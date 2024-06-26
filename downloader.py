from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bertin-project/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
