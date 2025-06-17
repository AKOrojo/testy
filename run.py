import torch
from transformers import AutoTokenizer

# Import the specific model class from your main.py file
from main import T5ForConditionalGeneration

# --- DIAGNOSTICS: Check for GPU availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The model name to load
model_name = "melmoth/ru-rope-t5-small-instruct"

# 1. Load the tokenizer and the model
print(f"Loading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model '{model_name}'...")
# Use the T5ForConditionalGeneration class from your main.py
model = T5ForConditionalGeneration.from_pretrained(model_name)
# --- Move model to the selected device ---
model.to(device)
print("Model loaded successfully.")

# 2. Prepare the input text in Russian
input_text = "<|user|>\nНапиши краткое содержание для рассказа 'Каштанка' Антона Чехова.<|endoftext|>\n<|assistant|>"

# 3. Tokenize the input text and move it to the selected device
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 4. Generate the output (inference)
print("Generating response...")
try:
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id # Explicitly set the end-of-sequence token
    )
    print("Generation complete.")

    # 5. Decode the output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 6. Print the result
    print("\n" + "="*40)
    print("Input Prompt:")
    print(input_text)
    print("\n" + "="*40)
    print("Generated Response:")
    print(full_output)

except Exception as e:
    print(f"An error occurred during generation: {e}")