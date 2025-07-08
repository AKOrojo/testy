import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import numpy as np
import evaluate

# --- Configuration ---
MODEL_PATH = "./c4-t5-from-scratch-stream"  # Path to your trained model
RESULTS_FILE = "evaluation_results.json"
ZERO_SHOT_RESULTS_DIR = "./zero_shot_results"
FINETUNED_RESULTS_DIR = "./finetuned_results"

# Ensure results directories exist
os.makedirs(ZERO_SHOT_RESULTS_DIR, exist_ok=True)
os.makedirs(FINETUNED_RESULTS_DIR, exist_ok=True)

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# --- Benchmark Definitions ---
BENCHMARKS = {
    "glue": ["mrpc", "cola", "stsb"],
    "cnndm": "cnn_dailymail",
    "squad": "squad",
    "sglue": ["copa", "wic"],
    "wmt": {
        "en-de": "wmt14",
        "en-fr": "wmt14",
        "en-ro": "wmt16",
    },
}

# --- Evaluation Functions ---

def evaluate_glue(model, tokenizer, subset, zero_shot=True):
    """Evaluates the model on a GLUE subset."""
    print(f"\n--- Evaluating GLUE:{subset} ({'Zero-Shot' if zero_shot else 'Fine-Tuned'}) ---")
    dataset = load_dataset("glue", subset)
    metric = evaluate.load("glue", subset)
    
    def preprocess_function(examples):
        if subset == "stsb":
            inputs = [f"{ex['sentence1']} {ex['sentence2']}" for ex in examples]
        else:
            inputs = [f"{ex['sentence1']} question: {ex['sentence2']}" for ex in examples]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        
        with tokenizer.as_target_tokenizer():
            if subset == "stsb":
                labels = tokenizer(
                    [str(label) for label in examples["label"]],
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                )
            else:
                labels = tokenizer(
                    [str(label > 0.5) for label in examples["label"]],
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    output_dir = f"{ZERO_SHOT_RESULTS_DIR}/glue_{subset}" if zero_shot else f"{FINETUNED_RESULTS_DIR}/glue_{subset}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    eval_results = trainer.evaluate()
    return {f"glue_{subset}": eval_results}

def evaluate_cnndm(model, tokenizer, zero_shot=True):
    """Evaluates the model on CNN/DailyMail."""
    print(f"\n--- Evaluating CNNDM ({'Zero-Shot' if zero_shot else 'Fine-Tuned'}) ---")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    rouge = evaluate.load("rouge")

    def preprocess_function(examples):
        inputs = [f"summarize: {doc}" for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    output_dir = f"{ZERO_SHOT_RESULTS_DIR}/cnndm" if zero_shot else f"{FINETUNED_RESULTS_DIR}/cnndm"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    eval_preds = trainer.predict(tokenized_dataset)
    decoded_preds = tokenizer.batch_decode(eval_preds.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(eval_preds.label_ids, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {"cnndm": result}

def evaluate_squad(model, tokenizer, zero_shot=True):
    """Evaluates the model on SQuAD."""
    print(f"\n--- Evaluating SQuAD ({'Zero-Shot' if zero_shot else 'Fine-Tuned'}) ---")
    dataset = load_dataset("squad", split="validation")
    squad_metric = evaluate.load("squad")

    def preprocess_function(examples):
        inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
        model_inputs = tokenizer(inputs, max_length=384, truncation=True, padding="max_length")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                [a["text"][0] if a["text"] else "" for a in examples["answers"]],
                max_length=32,
                truncation=True,
                padding="max_length",
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    output_dir = f"{ZERO_SHOT_RESULTS_DIR}/squad" if zero_shot else f"{FINETUNED_RESULTS_DIR}/squad"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    squad_preds = trainer.predict(tokenized_dataset)
    decoded_preds = tokenizer.batch_decode(squad_preds.predictions, skip_special_tokens=True)
    
    formatted_predictions = [{"id": ex["id"], "prediction_text": pred} for ex, pred in zip(dataset, decoded_preds)]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]

    result = squad_metric.compute(predictions=formatted_predictions, references=references)
    return {"squad": result}

def main():
    """Main function to run evaluations."""
    all_results = {"zero_shot": {}, "fine_tuned": {}}

    # --- Zero-Shot Evaluation ---
    print("--- Running Zero-Shot Evaluations ---")
    
    # GLUE
    for subset in BENCHMARKS["glue"]:
        all_results["zero_shot"].update(evaluate_glue(model, tokenizer, subset, zero_shot=True))

    # CNNDM
    all_results["zero_shot"].update(evaluate_cnndm(model, tokenizer, zero_shot=True))

    # SQuAD
    all_results["zero_shot"].update(evaluate_squad(model, tokenizer, zero_shot=True))
    
    # --- Fine-Tuning and Evaluation ---
    print("\n\n--- Running Fine-Tuning and Evaluations ---")

    # Fine-tune and evaluate on GLUE
    for subset in BENCHMARKS["glue"]:
        fine_tuned_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        all_results["fine_tuned"].update(evaluate_glue(fine_tuned_model, tokenizer, subset, zero_shot=False))

    # Fine-tune and evaluate on CNNDM
    fine_tuned_model_cnndm = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    all_results["fine_tuned"].update(evaluate_cnndm(fine_tuned_model_cnndm, tokenizer, zero_shot=False))

    # Fine-tune and evaluate on SQuAD
    fine_tuned_model_squad = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    all_results["fine_tuned"].update(evaluate_squad(fine_tuned_model_squad, tokenizer, zero_shot=False))

    # --- Save Final Results ---
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n\nEvaluation complete. Results saved to {RESULTS_FILE}")
    print(json.dumps(all_results, indent=4))

if __name__ == "__main__":
    main()