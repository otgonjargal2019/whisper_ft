"""
Fine-tune Whisper MEDIUM model with LoRA - OPTIMIZED for 6GB GPU
This version uses aggressive memory optimization techniques
"""
from pathlib import Path
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import soundfile as sf
import librosa
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import gc

FOOD_DATA = Path("food_data")

# Clear GPU memory before starting
torch.cuda.empty_cache()
gc.collect()

# Dataset
dataset = load_dataset("csv", data_files={
    "train": str(FOOD_DATA / "train.csv"),
    "validation": str(FOOD_DATA / "test.csv")
})

# Processor + Model - MEDIUM MODEL with 8-bit quantization
model_name = "openai/whisper-medium"
print(f"Loading {model_name} with 8-bit quantization for low memory...")
processor = WhisperProcessor.from_pretrained(model_name)

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# Load model with 8-bit quantization to save memory
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA - Smaller configuration for 6GB GPU
lora_config = LoraConfig(
    r=8,  # Reduced from 32 to save memory
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Only 2 modules to save memory
    lora_dropout=0.1,
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

# Training arguments - OPTIMIZED for 6GB GPU
training_args = Seq2SeqTrainingArguments(
    output_dir="fine_tuned_food_medium",
    per_device_train_batch_size=1,  # Batch size 1
    gradient_accumulation_steps=4,  # Simulate batch size 4
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    learning_rate=1e-4,  # Lower learning rate for stability with 8-bit
    warmup_steps=20,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=False,  # Disable FP16 - causes NaN with 8-bit quantization
    push_to_hub=False,
    gradient_checkpointing=True,  # Enable gradient checkpointing
    predict_with_generate=False,
    generation_max_length=225,
    logging_first_step=True,
    report_to=[],
    optim="adamw_8bit",  # Use 8-bit optimizer (not bnb version)
    max_grad_norm=0.3,  # Gradient clipping to prevent NaN
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

print("\nStarting training with Whisper Medium (8-bit quantized) model...")
print("Optimized for 6GB GPU with aggressive memory saving techniques!")
trainer.train()

print("\n✅ Training complete!")
print(f"Model saved to: fine_tuned_food_medium/")
