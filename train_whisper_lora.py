# from pathlib import Path
# from datasets import load_dataset
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
# from peft import LoraConfig, get_peft_model
# import soundfile as sf
# import librosa  

# FOOD_DATA = Path("food_data") 

# # Датаг ачаалах
# dataset = load_dataset("csv", data_files={"train": "food_data/train.csv", "validation": "food_data/test.csv"})

# # Модель + processor
# model_name = "openai/whisper-small"
# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)

# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=processor.tokenizer,  # зөвхөн tokenizer зааж өгнө
#     padding=True
# )

# # LoRA тохиргоо
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj","v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="SEQ_2_SEQ_LM"
# )
# model = get_peft_model(model, lora_config)

# # Datapreprocessing (simple)
# # def preprocess(batch):
# #     audio = batch["path"]
# #     batch["input_features"] = processor(audio, sampling_rate=16000).input_features
# #     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
# #     return batch

# def preprocess(batch):
#     audio_path = FOOD_DATA / batch["path"]
#     audio, sr = sf.read(audio_path)
#     if sr != 16000:
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
#     # processor-д padding=True зааж өгч байна
#     batch["input_features"] = processor(
#         audio, sampling_rate=16000, padding=True
#     ).input_features
#     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
#     return batch


# dataset = dataset.map(preprocess)

# # Training аргумент
# training_args = Seq2SeqTrainingArguments(
#     output_dir="fine_tuned_food",
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     learning_rate=3e-5,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     push_to_hub=False
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     tokenizer=processor.tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()


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

# Processor + Model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
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

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="fine_tuned_food",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-3,  # Higher learning rate for LoRA
    warmup_steps=50,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=False,
    push_to_hub=False,
    gradient_checkpointing=False,  # Disable gradient checkpointing - can cause issues with PEFT
    predict_with_generate=False,
    generation_max_length=225,
    logging_first_step=True,
    report_to=[],  # Disable wandb/tensorboard
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

trainer.train()



