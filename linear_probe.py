import os
import torch as torch
from torch import Tensor
from langdetect import detect
from langdetect import DetectorFactory
import logging
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from transformers.utils import is_bitsandbytes_available
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

print(f"Model: {MODEL_NAME}")
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")

#use langdetect to filter out all non-english examples
def is_english(text: str) -> bool:
    #Check if text is in English
    try:
        if isinstance(text, str) and len(text.strip()) > 0:
            return detect(text) == 'en'
        return False
    except:
        return False
    
#load healthbench 
DATASET = (load_dataset("openai/healthbench", data_files = "2025-05-07-06-14-12_oss_eval.jsonl"))
#filter out non-english entries

print("before filter:", len(DATASET["train"]))

# Debug: inspect schema and one example
# print("columns:", DATASET["train"].column_names)
# print("sample:", DATASET["train"][0])

def to_text(x):
    v = x.get("prompt", "")
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(str(i) for i in v)
    if isinstance(v, dict):
        return " ".join(str(i) for i in v.values())
    return str(v)

DATASET = DATASET.map(lambda x: {"text_for_langdetect": to_text(x)})

# Optional: avoid very short strings (langdetect is noisy on short text)
DATASET = DATASET.filter(lambda x: len(x["text_for_langdetect"].strip()) > 20)
DATASET = DATASET.filter(lambda x: is_english(x["text_for_langdetect"]))

print("after filter:", len(DATASET["train"]))

split = DATASET["train"].train_test_split(test_size=0.2, seed=42)
TRAIN_DATASET = split["train"]
TEST_DATASET = split["test"]


# extract activations from all layers at once
def extract_all_layers_pooled(model, tokenizer, data, batch_size, max_length=512):
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

#we will want this to be from the last layer, possibly 32 idfk
""" def extract_activations(model: AutoModel, tokenizer: AutoTokenizer, layer_num: int, data: list[str], batch_size: int) -> torch.Tensor:
    logging.info(f'Getting embeddings from layer {layer_num} for {len(data)} samples...')

    print(f'{len(data)}, and {batch_size}')
    max_length=512
    batch_num = 1
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        logging.debug(f'getting embeddings for batch {batch_num}....')
        batch_num += 1

        inputs = tokenizer(
            batch, return_tensors='pt', padding='max_length',truncation=True, max_length=max_length).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        embeddings = outputs.hidden_states[layer_num].cpu()
        logging.debug(f'Extracted hidden states of shape {embeddings.shape}')

        if i== 0:
            all_embeddings = embeddings
        else:
            all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)
        
    # logging.info(f'got embeddings for {len(data)} samples from layer {layer_num} with shape {all_embeddings.shape}')
    return all_embeddings """


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


def get_steering_direction(probe: Probe) ->  torch.Tensor:
    weight =  probe.probe[0].weight
    direction = weight[1] - weight[0]
    direction = direction / direction.norm()

    return direction.detach()

# use keywords to determine redirection
def label_from_rubrics(example):
    rubrics = example["rubrics"]
    redirect_keywords = ["consult", "see a doctor", "seek medical", "healthcare provider", 
                        "physician", "refer", "emergency", "professional"]
    rubric_text = " ".join([r["criterion"].lower() for r in rubrics])
    return 1 if any(kw in rubric_text for kw in redirect_keywords) else 0

train_labels = torch.tensor([label_from_rubrics(x) for x in split["train"]], dtype=torch.long)
print("Train:", torch.bincount(train_labels))

test_labels = torch.tensor(
    [label_from_rubrics(x) for x in split["test"]], 
    dtype=torch.long
)

def format_as_chat(text):
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


formatted_train = [format_as_chat(t) for t in split['train']['text_for_langdetect']]
formatted_test  = [format_as_chat(t) for t in split['test']['text_for_langdetect']]

if os.path.exists('train_hidden.pt') and os.path.exists('test_hidden.pt'): # if embeddings already extracted, use them

    print("Loading cached activations...")
    train_hidden = torch.load('train_hidden.pt', weights_only=False)
    test_hidden = torch.load('test_hidden.pt', weights_only=False)
    print("Loaded activations")
else:
    print("Extracting train activations...")
    train_hidden = extract_all_layers_pooled(model, tokenizer, formatted_train, batch_size=2)

    print("Extracting test activations...")
    test_hidden  = extract_all_layers_pooled(model, tokenizer, formatted_test,  batch_size=2)

    # store activations
    torch.save(train_hidden, 'train_hidden.pt')
    torch.save(test_hidden, 'test_hidden.pt')
    print("Saved activations")

pos_indices = np.where(np.array(train_labels) == 1)[0]
neg_indices = np.where(np.array(train_labels) == 0)[0]

# under-sample positives
pos_sampled = np.random.choice(pos_indices, size=len(neg_indices) * 2, replace=False)
balanced_indices = np.concatenate([pos_sampled, neg_indices])
np.random.shuffle(balanced_indices)
balanced_labels = torch.tensor(np.array(train_labels)[balanced_indices], dtype=torch.long)

layer_wise_accuracies = []
best_probe, best_layer, best_accuracy = None, -1, 0
batch_size = 2 #?

for layer_num in range(1):

    probe = Probe()
    
    probe.train(train_hidden[28][balanced_indices], balanced_labels, num_epoch=50, learning_rate=0.001) # train more idk

    accuracy, f1 = probe.evaluate(test_hidden[28], test_labels) # evaluate using f1 instead bc unbalanced dataset

    layer_wise_accuracies.append((accuracy,f1))

    if f1 > best_accuracy:
        best_probe, best_layer, best_accuracy = probe, 28, f1
    print(f"Layer {28}: {accuracy:.4f} | F1={f1:.4f}")


print(f"Best: {best_accuracy*100:.2f}% at layer {best_layer}")

steering_vector = get_steering_direction(best_probe)
hook = Hook(steering_vector = steering_vector, layer = best_layer, steering_coef = 25.0, apply_to_all_tokens=False)

messages = [
    {"role": "user", "content": "My elderly mum has been confused since this morning. Should I be worried?"}
]

# apply chat template
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # adds the assistant turn opener
)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

#generate steered output
hook.enable(model)
try:
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
        )
finally:
    hook.disable() 

print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

#  compare against unsteered output
with torch.no_grad():
    unsteered = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True, 
    pad_token_id=tokenizer.eos_token_id,          
    temperature=0.7,          
    repetition_penalty=1.3, 
)
print(tokenizer.decode(unsteered[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))