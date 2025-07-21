import argparse

import jiwer
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_model_and_processor(
    model_path, cache_dir="./cache", processor_from_parent_folder=False
):
    """Load the model and processor from the specified path."""
    print(f"Loading model from: {model_path}")

    processor = WhisperProcessor.from_pretrained(
        f"{model_path}/../" if processor_from_parent_folder else model_path,
        language="vietnamese",
        task="transcribe",
        cache_dir=cache_dir,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, cache_dir=cache_dir
    )

    return model, processor


def load_test_dataset(num_samples=500, cache_dir="./cache"):
    """Load the test dataset."""
    print(f"Loading test dataset with {num_samples} samples...")

    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "vi",
        split="test",
        cache_dir=cache_dir,
    ).shuffle(seed=42)
    dataset = dataset.select(range(num_samples)) if num_samples > 0 else dataset

    # Remove unnecessary columns
    dataset = dataset.remove_columns(
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

    # Cast audio to correct sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def calculate_wer_batch(model, processor, dataset, batch_size=8, device="cuda"):
    """Calculate WER using batch processing for efficiency."""
    print(f"Calculating WER on {len(dataset)} samples...")

    model.to(device)
    model.eval()

    predictions = []
    references = []

    # Process samples by batch
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))

        try:
            inputs = processor.feature_extractor(
                [b["array"] for b in batch["audio"]],
                sampling_rate=16000,
                return_tensors="pt",
            ).to(device)

            # Generate predictions
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    max_length=225,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            # Decode predictions
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            # Append to predictions and references
            predictions.extend(transcription)
            references.extend(batch["sentence"])
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Skip problematic samples
            continue

    # Calculate WER using jiwer
    wer_score = jiwer.wer(references, predictions)
    wer_percentage = wer_score * 100

    return wer_percentage, predictions, references


def print_sample_results(predictions, references, num_samples=5):
    """Print sample predictions vs references."""
    print(f"\nSample Results (showing first {num_samples}):")
    print("=" * 80)

    for i in range(min(num_samples, len(predictions))):
        print(f"Sample {i + 1}:")
        print(f"  Reference: '{references[i]}'")
        print(f"  Prediction: '{predictions[i]}'")
        print()


def main():
    parser = argparse.ArgumentParser(description="Calculate WER for Whisper model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (local path or Hugging Face model ID)",
    )
    parser.add_argument(
        "--processor_from_parent_folder",
        type=bool,
        default=False,
        help="Whether to load the processor from the parent folder of the local model path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of test samples to evaluate, -1 for all",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for datasets and models",
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=5,
        help="Number of sample predictions to show",
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load test dataset
    test_dataset = load_test_dataset(args.num_samples, args.cache_dir)

    print(f"Dataset loaded: {len(test_dataset)} samples")

    # Calculate WER using batch method
    model, processor = load_model_and_processor(
        args.model_path, args.cache_dir, args.processor_from_parent_folder
    )
    wer, predictions, references = calculate_wer_batch(
        model, processor, test_dataset, args.batch_size, args.device
    )

    # Print results
    print("\nWER RESULTS:")
    print(f"Model: {args.model_path}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Word Error Rate: {wer:.2f}%")

    # Show sample predictions
    if args.show_samples > 0:
        print_sample_results(predictions, references, args.show_samples)


if __name__ == "__main__":
    main()
