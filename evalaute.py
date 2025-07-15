import json
import os
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from main import T5ForConditionalGeneration

# --- Configuration ---
MODEL_PATH = "./c4-t5-from-scratch"
RESULTS_FILE = "evaluation_results.json"
ZERO_SHOT_RESULTS_DIR = "./zero_shot_results"
FINETUNED_RESULTS_DIR = "./finetuned_results"

# Ensure results directories exist
os.makedirs(ZERO_SHOT_RESULTS_DIR, exist_ok=True)
os.makedirs(FINETUNED_RESULTS_DIR, exist_ok=True)

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Add special tokens if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Benchmark Definitions ---
BENCHMARKS = {
    "glue": ["mrpc", "cola", "stsb"],
    "cnndm": "cnn_dailymail",
    "squad": "squad",
}

# --- Debug mode ---
DEBUG = True  # Set to True to see model predictions


# --- Evaluation Functions ---

def evaluate_glue(model, tokenizer, subset, zero_shot=True, num_samples=100):
    """Evaluates the model on a GLUE subset."""
    print(f"\n--- Evaluating GLUE:{subset} ({'Zero-Shot' if zero_shot else 'Fine-Tuned'}) ---")
    dataset = load_dataset("glue", subset)

    # Use a smaller subset for debugging
    if DEBUG and num_samples:
        dataset["validation"] = dataset["validation"].select(range(min(num_samples, len(dataset["validation"]))))

    metric = evaluate.load("glue", subset)

    # Map GLUE labels to text for T5
    if subset == "cola":
        label_map = {0: "unacceptable", 1: "acceptable"}
        task_prefix = "cola sentence: "
    elif subset == "mrpc":
        label_map = {0: "not_equivalent", 1: "equivalent"}
        task_prefix = "mrpc sentence1: "
    elif subset == "stsb":
        # For regression tasks, we'll discretize into buckets
        def map_stsb_label(score):
            # Convert 0-5 scale to text categories
            if score < 1:
                return "0"
            elif score < 2:
                return "1"
            elif score < 3:
                return "2"
            elif score < 4:
                return "3"
            elif score < 5:
                return "4"
            else:
                return "5"

        label_map = map_stsb_label
        task_prefix = "stsb sentence1: "
    else:
        label_map = {0: "no", 1: "yes"}
        task_prefix = f"{subset} sentence1: "

    def preprocess_function(examples):
        # Format inputs for T5 text-to-text
        if subset == "cola":
            # Single sentence acceptability
            inputs = [f"{task_prefix}{s}" for s in examples['sentence']]
        elif subset in ["mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]:
            # Sentence pair tasks
            inputs = [f"{task_prefix}{s1} sentence2: {s2}"
                      for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
        else:
            # Handle other formats
            inputs = examples['sentence'] if 'sentence' in examples else examples['text']

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Format labels as text
        if subset == "stsb":
            # For STS-B, use the mapping function
            text_labels = [label_map(label) for label in examples['label']]
        else:
            # For classification tasks
            text_labels = [label_map[int(label)] for label in examples['label']]

        # Tokenize labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                text_labels,
                max_length=32,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

        # Replace padding token id with -100 so it's ignored in loss
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels["input_ids"]

        # Store original labels for debugging
        model_inputs["original_labels"] = text_labels

        return model_inputs

    # Process dataset
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    )

    # Add back the original numeric labels for metric computation
    if "validation" in dataset:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].add_column(
            "numeric_labels",
            dataset["validation"]["label"]
        )

    output_dir = f"{ZERO_SHOT_RESULTS_DIR}/glue_{subset}" if zero_shot else f"{FINETUNED_RESULTS_DIR}/glue_{subset}"

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=10,  # Short for classification
        generation_num_beams=1,
        fp16=False,  # Disable for debugging
        remove_unused_columns=False,
    )

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        if DEBUG:
            print(f"\nFirst 5 predictions for {subset}:")
            for i in range(min(5, len(decoded_preds))):
                print(f"  Pred: '{decoded_preds[i]}'")

        # Map predictions back to numeric labels
        if subset == "cola":
            pred_map = {"unacceptable": 0, "acceptable": 1}
        elif subset == "mrpc":
            pred_map = {"not_equivalent": 0, "equivalent": 1, "not": 0, "equivalent": 1}
        elif subset == "stsb":
            # For STS-B, map back to numeric scores
            pred_map = {str(i): float(i) for i in range(6)}
        else:
            pred_map = {"no": 0, "yes": 1, "false": 0, "true": 1}

        # Convert predictions to numeric
        numeric_preds = []
        for pred in decoded_preds:
            pred = pred.strip().lower()
            if pred in pred_map:
                numeric_preds.append(pred_map[pred])
            else:
                # Default prediction if model generates unexpected text
                if subset == "stsb":
                    numeric_preds.append(2.5)  # Middle value
                else:
                    numeric_preds.append(0)  # Default to negative class

        # Get true numeric labels from the dataset
        numeric_labels = eval_preds.label_ids if hasattr(eval_preds, 'label_ids') else labels

        # For GLUE tasks, we stored numeric labels separately
        if hasattr(trainer.eval_dataset, 'numeric_labels'):
            numeric_labels = trainer.eval_dataset['numeric_labels']

        # Ensure same length
        min_len = min(len(numeric_preds), len(numeric_labels))
        numeric_preds = numeric_preds[:min_len]
        numeric_labels = list(numeric_labels[:min_len])

        if DEBUG:
            print(f"Numeric predictions: {numeric_preds[:5]}")
            print(f"Numeric labels: {numeric_labels[:5]}")

        # Compute metrics
        if subset == "stsb":
            # Ensure float arrays for correlation
            numeric_preds = np.array(numeric_preds, dtype=float)
            numeric_labels = np.array(numeric_labels, dtype=float)

            # Add small noise to avoid constant arrays
            if len(set(numeric_preds)) == 1:
                numeric_preds = numeric_preds + np.random.normal(0, 0.01, size=len(numeric_preds))

            result = metric.compute(predictions=numeric_preds, references=numeric_labels)
        else:
            # For classification tasks
            result = metric.compute(predictions=numeric_preds, references=numeric_labels)

        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    return {f"glue_{subset}": eval_results}


def evaluate_squad(model, tokenizer, zero_shot=True, num_samples=100):
    """Evaluates the model on SQuAD."""
    print(f"\n--- Evaluating SQuAD ({'Zero-Shot' if zero_shot else 'Fine-Tuned'}) ---")
    dataset = load_dataset("squad", split=f"validation[:{num_samples}]" if DEBUG else "validation")
    squad_metric = evaluate.load("squad")

    def preprocess_function(examples):
        inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
        model_inputs = tokenizer(
            inputs,
            max_length=384,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Extract answer text
        answers = []
        for answer_dict in examples["answers"]:
            if answer_dict["text"]:
                answers.append(answer_dict["text"][0])
            else:
                answers.append("")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                answers,
                max_length=32,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

        # Replace padding token id with -100
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Keep original data for metric computation
    tokenized_dataset = tokenized_dataset.add_column("id", dataset["id"])
    tokenized_dataset = tokenized_dataset.add_column("answers", dataset["answers"])

    output_dir = f"{ZERO_SHOT_RESULTS_DIR}/squad" if zero_shot else f"{FINETUNED_RESULTS_DIR}/squad"

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=32,
        generation_num_beams=1,  # Start with greedy decoding
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    predictions = trainer.predict(tokenized_dataset)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

    if DEBUG:
        print("\nFirst 5 SQuAD predictions:")
        for i in range(min(5, len(decoded_preds))):
            print(f"  Question: {dataset[i]['question']}")
            print(f"  Predicted: '{decoded_preds[i]}'")
            print(f"  Actual: '{dataset[i]['answers']['text'][0] if dataset[i]['answers']['text'] else 'No answer'}'")
            print()

    # Format predictions for SQuAD metric
    formatted_predictions = [
        {"id": id_, "prediction_text": pred.strip()}
        for id_, pred in zip(tokenized_dataset["id"], decoded_preds)
    ]
    references = [
        {"id": id_, "answers": answers}
        for id_, answers in zip(tokenized_dataset["id"], tokenized_dataset["answers"])
    ]

    result = squad_metric.compute(predictions=formatted_predictions, references=references)
    return {"squad": result}


def simple_generation_test(model, tokenizer):
    """Test basic generation capability of the model."""
    print("\n--- Testing Basic Generation ---")
    test_inputs = [
        "translate English to French: Hello, how are you?",
        "summarize: The quick brown fox jumps over the lazy dog.",
        "cola sentence: This is a good sentence.",
        "mrpc sentence1: The cat sat on the mat. sentence2: A cat was sitting on a mat.",
    ]

    for test_input in test_inputs:
        inputs = tokenizer(test_input, return_tensors="pt", padding=True)
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_input}")
        print(f"Output: '{decoded}'")
        print()


def main():
    """Main function to run evaluations."""
    all_results = {"zero_shot": {}, "fine_tuned": {}}

    # First, test basic generation
    if DEBUG:
        simple_generation_test(model, tokenizer)

    # --- Zero-Shot Evaluation ---
    print("\n--- Running Zero-Shot Evaluations ---")

    # GLUE
    for subset in BENCHMARKS["glue"]:
        try:
            result = evaluate_glue(model, tokenizer, subset, zero_shot=True)
            all_results["zero_shot"].update(result)
        except Exception as e:
            print(f"Error evaluating GLUE {subset}: {e}")
            import traceback
            traceback.print_exc()
            all_results["zero_shot"][f"glue_{subset}"] = {"error": str(e)}

    # SQuAD
    try:
        result = evaluate_squad(model, tokenizer, zero_shot=True)
        all_results["zero_shot"].update(result)
    except Exception as e:
        print(f"Error evaluating SQuAD: {e}")
        import traceback
        traceback.print_exc()
        all_results["zero_shot"]["squad"] = {"error": str(e)}

    # --- Save Results ---
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n\nEvaluation complete. Results saved to {RESULTS_FILE}")

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    for eval_type, results in all_results.items():
        if results:
            print(f"\n{eval_type.upper()}:")
            for task, metrics in results.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    print(f"  {task}:")
                    for metric, value in metrics.items():
                        if metric.startswith("eval_"):
                            print(f"    {metric}: {value:.4f}" if isinstance(value, (int,
                                                                                     float)) else f"    {metric}: {value}")


if __name__ == "__main__":
    main()