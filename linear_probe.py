import os
import torch as torch
from torch import Tensor
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

login(token = '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')


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

# PROBE_LAYER = 14
# INTERVENE_LAYER = 8

print(f"Model: {MODEL_NAME}")
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")

DATASET = (load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
        .train_test_split(train_size=800, test_size=200))

# DATASET_NAMES = ["healthbench"] #Bitching abt healthbench to do with the inconsistent columns


#we will want this to be from the last layer, possibly 32 idfk
def extract_activations(model: AutoModel, tokenizer: AutoTokenizer, layer_num: int, data: list[str], batch_size: int) -> torch.Tensor:
    logging.info(f'Getting embeddings from layer {layer_num} for {len(data)} samples...')
    
    batch_num = 1
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        logging.debug(f'getting embeddings for batch {batch_num}....')
        batch_num += 1

        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        embeddings = outputs.hidden_states[layer_num]
        logging.debug(f'Extracted hidden states of shape {embeddings.shape}')

        if i== 0:
            all_embeddings = embeddings
        else:
            all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)
        
    logging.info(f'got embeddings for {len(data)} samples from layer {layer_num} with shape {all_embeddings.shape}')
    return all_embeddings
        
        
class Probe():
    def __init__(self, hidden_dim: int = 4096, class_size: int = 2) -> None:
        self.probe = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, class_size),
            torch.nn.Sigmoid()
        )
    
    def train(self, data_embeddings: torch.Tensor, labels: torch.Tensor, num_epoch: int = 3,
              learning_rate: float = 0.001, batch_size : int = 32) -> None:
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.probe.parameters(), lr=learning_rate)

        logging.info("training the probe...")
        for epoch in range(num_epoch):
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size].detatch()
                batch_labels = labels[i:i+batch_size]

                batch_embeddings = torch.mean(batch_embeddings, dim=1)

                outputs = self.probe(batch_embeddings)

                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        logging.info('Done.')

    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        for i in range(0, len(data_embeddings), batch_size):
            batch_embeddings = data_embeddings[i:i+batch_size]

            batch_embeddings = torch.mean(batch_embeddings, dim=1)

            outputs = self.probe(batch_embeddings)

            _, predicted = torch.max(outputs, 1)

            #concatenate the predictions from each batch
            if i == 0:
                all_predicted = predicted
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)

        return all_predicted
    
    def evaluate(self, data_embeddings: torch.tensor, labels: torch.tensor, batch_size: 32) -> float:
        for i in range(0, len(data_embeddings), batch_size):

            batch_embeddings = data_embeddings[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batch_embeddings = torch.mean(batch_embeddings, dim=1)

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