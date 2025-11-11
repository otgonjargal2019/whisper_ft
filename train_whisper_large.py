"""
Fine-tune Whisper LARGE-V3 model with LoRA for Mongolian food detection
This uses the largest model (1.5B parameters) for best accuracy
WARNING: Requires significant GPU memory (16GB+ recommended)
"""
from pathlib import Path
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import soundfile as sf
import librosa
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

FOOD_DATA = Path("food_data")

# Dataset
dataset = load_dataset("csv", data_files={
    "train": str(FOOD_DATA / "train.csv"),
    "validation": str(FOOD_DATA / "test.csv")
})

# Processor + Model - LARGE-V3 MODEL
model_name = "openai/whisper-large-v3"
print(f"Loading {model_name}...")
print("⚠️  This is a large model and will take time to download and load...")
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

# LoRA - Using even larger rank for large model
lora_config = LoraConfig(
    r=64,  # Large rank for large model
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,  # Lower dropout for large model
    bias="none",
)
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Preprocess
def preprocess(batch):
    audio_path = FOOD_DATA / batch["path"]
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=["path", "text"])

# Data collator for Whisper
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": torch.tensor(feature["input_features"])} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Return only input_features and labels (remove any extra keys)
        return {"input_features": batch["input_features"], "labels": batch["labels"]}

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Training arguments - adjusted for large model
training_args = Seq2SeqTrainingArguments(
    output_dir="fine_tuned_food_large",  # Different output dir
    per_device_train_batch_size=1,  # Small batch size for large model
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,  # Simulate larger batch
    num_train_epochs=5,
    learning_rate=2e-4,  # Lower learning rate for stability
    warmup_steps=50,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=torch.cuda.is_available(),  # Use FP16 on GPU to save memory
    push_to_hub=False,
    gradient_checkpointing=True,  # Enable for large model to save memory
    predict_with_generate=False,
    generation_max_length=225,
    logging_first_step=True,
    report_to=[],
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

print("\nStarting training with Whisper Large-V3 model...")
print("This will take the longest but should give the best results!")
print("Make sure you have enough GPU memory (16GB+ recommended)")
trainer.train()

print("\n✅ Training complete!")
print(f"Model saved to: fine_tuned_food_large/")
