import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import login
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

warnings.filterwarnings("ignore")

login()

MODEL_NAME = "openai/whisper-base"
DATASET_PATH = "mozilla-foundation/common_voice_17_0"

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    MODEL_NAME, cache_dir="./cache"
)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language="vietnamese", task="transcribe", cache_dir="./cache"
)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir="./cache")
# Disable cache for gradient checkpointing compatibility
model.config.use_cache = False


def filter_audio_length(example):
    """Filter out audio samples that are too short (<1s) or too long (>30s)"""
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    duration = len(audio_array) / sampling_rate

    # Keep only audio between 1 and 30 seconds
    return 1.0 <= duration <= 30.0


def calculate_audio_hours(dataset_dict):
    """Calculate total audio hours for each split in the dataset."""
    print("Dataset statistics:")
    for split_name, dataset in dataset_dict.items():
        if "audio" in dataset.column_names:
            # For original dataset with audio column
            total_duration = sum(
                len(audio["array"]) / audio["sampling_rate"]
                for audio in dataset["audio"]
            )
        hours = total_duration / 3600
        print(
            f"{split_name.upper()} set: {len(dataset):,} samples, {hours:.2f} hours of audio"
        )


def prepare_dataset(batch):
    """Prepare the dataset for training."""
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def load_or_prepare_dataset(dataset_path=DATASET_PATH):
    """Check if prepared dataset exists, load it if it does, otherwise prepare and save it."""
    processed_dataset_path = dataset_path.replace("/", "_")
    if os.path.exists(f"{processed_dataset_path}_processed"):
        print(f"Loading prepared dataset from {processed_dataset_path}_processed")
        return DatasetDict.load_from_disk(f"{processed_dataset_path}_processed")
    else:
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(
            dataset_path,
            "vi",
            split="train+validation",
        )
        common_voice["test"] = load_dataset(
            dataset_path,
            "vi",
            split="test",
        )
        common_voice = common_voice.remove_columns(
            [
                "accent",
                "age",
                "client_id",
                "down_votes",
                "gender",
                "locale",
                "path",
                "segment",
                "up_votes",
                "variant",
            ]
        )

        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

        print("Filtering audio by length (1-30 seconds)...")
        common_voice = common_voice.filter(filter_audio_length)

        common_voice["train"] = (
            common_voice["train"].shuffle(seed=42).select(range(1000))
        )
        common_voice["test"] = common_voice["test"].shuffle(seed=42).select(range(500))
        calculate_audio_hours(common_voice)

        print("Preparing dataset...")
        prepared_dataset = common_voice.map(
            prepare_dataset,
            remove_columns=common_voice.column_names["train"],
            num_proc=5,
        )

        print(f"Saving prepared dataset to {processed_dataset_path}_processed")
        prepared_dataset.save_to_disk(f"{processed_dataset_path}_processed")

        return prepared_dataset


common_voice = load_or_prepare_dataset(DATASET_PATH)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")


def compute_metrics(pred):
    """Compute the metrics for the model."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{MODEL_NAME.replace('/', '_')}-vi",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=300,
    max_steps=1000,
    gradient_checkpointing=True,
    bf16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)


def evaluate_model(trainer, dataset_name="test"):
    """Evaluate the model and return metrics."""
    print(f"\n=== Evaluating on {dataset_name} set ===")
    eval_results = trainer.evaluate()
    wer = eval_results.get("eval_wer", 0)
    print(f"WER: {wer:.2f}%")
    return eval_results


# Evaluate before training
print("Evaluating model BEFORE training...")
initial_metrics = evaluate_model(trainer, "test")

# Train the model
print("Starting training...")
trainer.train()

# Evaluate after training
print("Evaluating model AFTER training...")
final_metrics = evaluate_model(trainer, "test")

# Show improvement
initial_wer = initial_metrics.get("eval_wer", 0)
final_wer = final_metrics.get("eval_wer", 0)
improvement = initial_wer - final_wer
improvement_pct = (improvement / initial_wer * 100) if initial_wer > 0 else 0

print("TRAINING RESULTS:")
print(f"Initial WER: {initial_wer:.2f}%")
print(f"Final WER: {final_wer:.2f}%")
print(
    f"Improvement: {improvement:.2f} percentage points ({improvement_pct:.1f}% relative improvement)"
)
