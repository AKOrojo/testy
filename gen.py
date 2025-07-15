# A simple script to test generation
import torch
from transformers import AutoTokenizer
from main import T5ForConditionalGeneration

# Load the model and tokenizer from your training output directory
model_path = "./c4-t5-from-scratch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# --- Test Prompts ---

prompt1 = "translate English to French: Hello, how are you?"
prompt2 = "The capital of the United States is"
prompt3 = "The best way to learn programming is to <extra_id_0> every day."

# --- Generate ---

for prompt in [prompt1, prompt2, prompt3]:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=50)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input: {prompt}")
    print(f"Likely Output: '{decoded_output}'")
    print("-" * 20)