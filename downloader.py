import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
model_name = "google/gemma-2b-it"

# Ensuring CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Updating configuration to avoid the hidden_act warning
config = AutoConfig.from_pretrained(model_name)
if hasattr(config, 'hidden_act'):
    config.hidden_activation = config.hidden_act
    del config.hidden_act

tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)
