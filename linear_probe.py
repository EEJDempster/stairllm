import os
import torch as torch
from torch import Tensor
from langdetect import detect
from langdetect import DetectorFactory
# import system
import logging
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # This is needed to avoid a warning from huggingface


login(token = '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')

DetectorFactory.seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
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


#we will want this to be from the last layer, possibly 32 idfk
def extract_activations(model: AutoModel, tokenizer: AutoTokenizer, layer_num: int, data: list[str], batch_size: int) -> torch.Tensor:
    logging.info(f'Getting embeddings from layer {layer_num} for {len(data)} samples...')

    print(f'{len(data)}, and {batch_size}')
    batch_num = 1
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        logging.debug(f'getting embeddings for batch {batch_num}....')
        batch_num += 1

        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        embeddings = outputs.hidden_states[layer_num].cpu()
        logging.debug(f'Extracted hidden states of shape {embeddings.shape}')

        if i== 0:
            all_embeddings = embeddings
        else:
            all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)
        
    # logging.info(f'got embeddings for {len(data)} samples from layer {layer_num} with shape {all_embeddings.shape}')
    return all_embeddings
        
        
class Probe():
    def __init__(self, hidden_dim: int = 4096, class_size: int = 2) -> None:
        self.probe = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, class_size),
            torch.nn.Sigmoid()
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
                batch_labels = labels[i:i+batch_size].to_device()

                batch_embeddings = torch.mean(batch_embeddings, dim=1).to(device)

                outputs = self.probe(batch_embeddings)

                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        logging.info('Done.')

    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        for i in range(0, len(data_embeddings), batch_size):
            batch_embeddings = data_embeddings[i:i+batch_size]

            batch_embeddings = torch.mean(batch_embeddings, dim=1).to(device)

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

            batch_embeddings = torch.mean(batch_embeddings, dim=1).to(device)

            with torch.no_grad():
                outputs = self.probe(batch_embeddings)

            _, predicted = torch.max(outputs, dim=-1)

            if i == 0:
                all_predicted = predicted
                all_labels = batch_labels
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)
                all_labels = torch.cat([all_labels, batch_labels], dim=0)
        
        #calculate accuracy
        correct = (all_predicted == all_labels).sum().item()
        accuracy = correct/all_labels.shape[0]
        logging.info('Probe accuracy = {accuracy:.2f}')
        return accuracy
    

target_tag = "theme:emergency_referrals" # label based on whether there is an emergency referral in response

train_labels = torch.tensor(
    [1 if target_tag in x["example_tags"] else 0 for x in split["train"]],
    dtype=torch.long
)
test_labels = torch.tensor(
    [1 if target_tag in x["example_tags"] else 0 for x in split["test"]],
    dtype=torch.long
)

layer_wise_accuracies = []
best_probe, best_layer, best_accuracy = None, -1, 0
batch_size = 32 #?

for layer_num in range(NUM_LAYERS):
    logging.info(f'evaluating representations of layer {layer_num}:\n')

    train_embeddings = extract_activations(model, tokenizer, layer_num=layer_num, data=split['train']['text_for_langdetect'], batch_size=batch_size)
    test_embeddings = extract_activations(model, tokenizer, layer_num=layer_num, data=split['test']['text_for_langdetect'], batch_size=batch_size)

    probe = Probe()
    probe.train(data_embeddings=train_embeddings, labels = train_labels, num_epoch=5, learning_rate=0.001, batch_size=8)
    accuracy = probe.evaluate(data_embeddings=test_embeddings, labels=test_labels)
    layer_wise_accuracies.append(accuracy)

    # Keep track of the best probe
    if accuracy > best_accuracy:
        best_probe, best_layer, best_accuracy = probe, layer_num, accuracy

logging.info(f'DONE.\n Best accuracy of {best_accuracy*100}% from layer {best_layer}.')