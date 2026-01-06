import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import Pipeline, AutoTokenizer

class BaseLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)