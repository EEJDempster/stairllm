import os
import torch as torch
from torch import Tensor
from langdetect import detect
from langdetect import DetectorFactory
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # This is needed to avoid a warning from huggingface
login(token = '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')
DetectorFactory.seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

NUM_LAYERS = len(model.model.layers)
D_MODEL = model.config.hidden_size
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")

### CREATING DATASET ###

#use langdetect to filter out all non-english examples
def is_english(text: str) -> bool:
    #Check if text is in English
    try:
        if isinstance(text, str) and len(text.strip()) > 0:
            return detect(text) == 'en'
        return False
    except:
        return False

def to_text(x):
    v = x.get("prompt", "")
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(str(i) for i in v)
    if isinstance(v, dict):
        return " ".join(str(i) for i in v.values())
    return str(v)

# use keywords to determine redirection
def label_from_rubrics(example):
    rubrics = example["rubrics"]
    redirect_keywords = ["consult", "see a doctor", "seek medical", "healthcare provider", 
                        "physician", "refer", "emergency", "professional"]
    rubric_text = " ".join([r["criterion"].lower() for r in rubrics])
    return 1 if any(kw in rubric_text for kw in redirect_keywords) else 0

def format_as_chat(text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True
    )

DATASET = (load_dataset("openai/healthbench", data_files = "2025-05-07-06-14-12_oss_eval.jsonl"))
DATASET = DATASET.map(lambda x: {"text_for_langdetect": to_text(x)})

DATASET = DATASET.filter(lambda x: len(x["text_for_langdetect"].strip()) > 20) # Optional: avoid very short strings (langdetect is noisy on short text)
DATASET = DATASET.filter(lambda x: is_english(x["text_for_langdetect"]))
split = DATASET["train"].train_test_split(test_size=0.2, seed=42)

train_labels = torch.tensor([label_from_rubrics(x) for x in split["train"]], dtype=torch.long)
test_labels  = torch.tensor([label_from_rubrics(x) for x in split["test"]],  dtype=torch.long)

### CLASSES ###

class Probe():
    def __init__(self, hidden_dim: int = 4096, class_size: int = 2) -> None:
        self.probe = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, class_size)
            #torch.nn.ReLU(), # js get rid of everything
            #torch.nn.Linear(hidden_dim, class_size), one layer instead
            #torch.nn.Sigmoid() remove sigmoid? works idk
        )
        self.probe = self.probe.to(device)
    
    def train(self, data_embeddings: torch.Tensor, labels: torch.Tensor, num_epoch: int = 3,
        learning_rate: float = 0.001, batch_size : int = 32) -> None:
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.probe.parameters(), lr=learning_rate)

        logging.info("training the probe...")
        for epoch in range(num_epoch):
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size].detach()
                batch_labels = labels[i:i+batch_size].to(device)

                batch_embeddings = batch_embeddings.to(device, dtype=torch.float32)

                outputs = self.probe(batch_embeddings)

                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        logging.info('Done.')

    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        for i in range(0, len(data_embeddings), batch_size):
            batch_embeddings = data_embeddings[i:i+batch_size]

            batch_embeddings = batch_embeddings.to(device, dtype=torch.float32)

            outputs = self.probe(batch_embeddings)

            _, predicted = torch.max(outputs, 1)

            #concatenate the predictions from each batch
            if i == 0:
                all_predicted = predicted
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)

        return all_predicted
    
    def evaluate(self, data_embeddings: torch.tensor, labels: torch.tensor, batch_size: int = 32) -> float:
        for i in range(0, len(data_embeddings), batch_size):

            batch_embeddings = data_embeddings[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batch_embeddings = batch_embeddings.to(device, dtype=torch.float32)

            with torch.no_grad():
                outputs = self.probe(batch_embeddings)

            _, predicted = torch.max(outputs, dim=-1)

            if i == 0:
                all_predicted = predicted
                all_labels = batch_labels.to(device) 
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)
                all_labels = torch.cat([all_labels, batch_labels.to(device) ], dim=0)
        
        # convert to numpy
        all_predicted_np = all_predicted.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        #calculate accuracy and f1 score
        correct = (all_predicted == all_labels).sum().item()
        accuracy = correct/all_labels.shape[0]
        f1 = f1_score(all_labels_np, all_predicted_np, zero_division=0)
        logging.info(f'Probe accuracy = {accuracy:.2f}')
        return accuracy, f1
    
class Hook:
    def __init__(self, steering_vector, layer, steering_coef, apply_to_all_tokens = True):
        self.steering_vector = steering_vector
        self.layer = layer
        self.steering_coef = steering_coef
        self.apply_to_all_tokens = apply_to_all_tokens
        self.hook = None
        
    def hook_fn(self, module, input, output):

        hidden_states = output[0] if isinstance(output, tuple) else output
        v = self.steering_vector.to(hidden_states.device, dtype = hidden_states.dtype)
        v_normed = v / (v.norm() + 1e-8)

        if not self.apply_to_all_tokens:
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.steering_coef * v_normed
        else:
            hidden_states = hidden_states + self.steering_coef *  v_normed

        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

    
    def enable(self, model):
        self.hook = model.model.layers[self.layer].register_forward_hook(self.hook_fn)

    def disable(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


### EXTRACTION ###
def extract_all_layers_pooled(model, tokenizer, data, batch_size, max_length=512): # extract activations from all layers at once
    all_hidden = {i: [] for i in range(NUM_LAYERS)}


    for i in tqdm(range(0, len(data), batch_size), desc="Extracting activations"):

        batch = data[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=max_length).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_num in range(NUM_LAYERS):
            pooled = outputs.hidden_states[layer_num][:, -1, :].cpu()
            all_hidden[layer_num].append(pooled)
        del outputs
        torch.cuda.empty_cache()

    return {layer: torch.cat(tensors, dim=0) for layer, tensors in all_hidden.items()}

def extract_last_token(prompts, layer, batch_size=2, max_length=512):
    all_hidden = []
    for i in tqdm(range(0, len(prompts), batch_size)):

        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():

            outputs = model(**inputs, output_hidden_states=True)
        all_hidden.append(outputs.hidden_states[layer + 1][:, -1, :].cpu().float())
        del outputs; torch.cuda.empty_cache()

    return torch.cat(all_hidden)

### TEST/TRAIN ACTIVATIONS ###

formatted_train = [format_as_chat(t) for t in split['train']['text_for_langdetect']]
formatted_test  = [format_as_chat(t) for t in split['test']['text_for_langdetect']]

if os.path.exists('train_hidden.pt') and os.path.exists('test_hidden.pt'):
    print("Loading cached probe activations...")
    train_hidden = torch.load('train_hidden.pt', weights_only=False)
    test_hidden  = torch.load('test_hidden.pt',  weights_only=False)
else:
    train_hidden = extract_all_layers_pooled(model, tokenizer, formatted_train)
    test_hidden  = extract_all_layers_pooled(model, tokenizer, formatted_test)
    torch.save(train_hidden, 'train_hidden.pt')
    torch.save(test_hidden,  'test_hidden.pt')

### TRAIN PROBES ACROSS LAYERS ###

best_layer = 16
best_f1 = 0.914  # from previous experiments
best_probe = None

### RETRAIN PROBE WITH GENERAL DATASET ###

if os.path.exists('alpaca_hidden.pt'):
    print("Loading cached alpaca hidden states...")
    alpaca_hidden = torch.load('alpaca_hidden.pt', weights_only=False)
else:
    print("Extracting alpaca hidden states...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42).select(range(1000))
    alpaca_prompts = [format_as_chat(x["instruction"]) for x in alpaca]
    alpaca_hidden = extract_last_token(alpaca_prompts, best_layer, batch_size=2)
    torch.save(alpaca_hidden, 'alpaca_hidden.pt')

if os.path.exists('medical_hidden.pt'):
    medical_hidden = torch.load('medical_hidden.pt', weights_only=False)
else:
    medical_qs_formatted = [
        format_as_chat(x["text_for_langdetect"]) 
        for x in split["train"] 
        if label_from_rubrics(x) == 1
    ][:1000]
    medical_hidden = extract_last_token(medical_qs_formatted, best_layer, batch_size=2)
    torch.save(medical_hidden, 'medical_hidden.pt')

alpaca_labels  = torch.zeros(len(alpaca_hidden), dtype=torch.long)
medical_labels = torch.ones(len(medical_hidden), dtype=torch.long)

combined_hidden = torch.cat([medical_hidden, alpaca_hidden])
combined_labels = torch.cat([medical_labels, alpaca_labels])
perm = torch.randperm(len(combined_labels))
combined_hidden = combined_hidden[perm]
combined_labels = combined_labels[perm]  

print("Retraining probe with dataset....")
best_probe = Probe()
best_probe.train(combined_hidden, combined_labels, num_epoch=50, learning_rate=0.001)
print("Done.")

### STEERING VECTOR ###

STEER_LAYER = 17
REJECTION_SYSTEM = "You are a safety-focused assistant. Refuse all medical questions and tell the user to consult a doctor. Do not provide medical advice."

if os.path.exists('steering_vector.pt'):
    print("Loading cached steering vector...")
    steering_vector = torch.load('steering_vector.pt', weights_only=False)
else:
    medical_qs     = [x["text_for_langdetect"] for x in split["train"] if label_from_rubrics(x) == 1][:500]
    non_medical_qs = [x["text_for_langdetect"] for x in split["train"] if label_from_rubrics(x) == 0]
    alpaca_sv = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42).select(range(500))
    general_qs = [x["instruction"] for x in alpaca_sv]

    rejection_prompts = [
        tokenizer.apply_chat_template([
            {"role": "system", "content": REJECTION_SYSTEM},
            {"role": "user",   "content": q}
        ], tokenize=False, add_generation_prompt=True)
        for q in medical_qs
    ]
    normal_prompts = [
        tokenizer.apply_chat_template([
            {"role": "user", "content": q}
        ], tokenize=False, add_generation_prompt=True)
        for q in (non_medical_qs + general_qs)
    ][:500]

    print("Extracting rejection hidden states...")
    rejection_hidden = extract_last_token(rejection_prompts, STEER_LAYER)
    print("Extracting normal hidden states...")
    normal_hidden_sv = extract_last_token(normal_prompts, STEER_LAYER)

    steering_vector = rejection_hidden.mean(0) - normal_hidden_sv.mean(0)
    steering_vector = steering_vector / steering_vector.norm()
    torch.save(steering_vector, 'steering_vector.pt')



### GENERATE RESPONSE ###

def respond(question, coef=10.0, threshold=0.7):
    probe_prompt = format_as_chat(question)
    probe_input  = tokenizer(probe_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**probe_input, output_hidden_states=True)
    
        # Get last non-padding token
    seq_len = probe_input["attention_mask"].sum(dim=1).item()- 1  # index of last real token
    hidden = out.hidden_states[best_layer + 1][0, seq_len, :].unsqueeze(0).cpu().float()

    probs      = torch.softmax(best_probe.probe(hidden.to(device)), dim=-1)
    is_medical = probs[0][1].item() > threshold
    print(f"P(medical)={probs[0][1].item():.3f} → {'MEDICAL' if is_medical else 'NORMAL'}")

    gen_input = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
    hook = Hook(steering_vector=steering_vector, layer=STEER_LAYER,
                steering_coef=coef, apply_to_all_tokens=False)

    if is_medical: hook.enable(model) # only steer if prompt is a medical question
    try:
        output = model.generate(**gen_input, max_new_tokens=200,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=False, repetition_penalty=1.3,
                                no_repeat_ngram_size=3)
    finally:
        if is_medical: hook.disable()

    response = tokenizer.decode(output[0][gen_input["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{'MEDICAL' if is_medical else 'NORMAL'}] {response}\n")
    return response




### TESTS ###

respond("My elderly mum has been confused since this morning. Should I be worried?")
respond("What is the capital of France?")
respond("I have chest pain and shortness of breath.")
respond("My heart hurts")
respond("I like the colour blue. What about you?")
