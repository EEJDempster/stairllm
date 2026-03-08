import os
import torch as t
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


device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16

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



def get_embeddings_from_model(model: AutoModel, tokenizer: AutoTokenizer, layer_num: int, data: list[str], batch_size: int) -> torch.Tensor:
    