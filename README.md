# Fine-tuned Whisper Model for Mongolian Food Detection

This project contains a fine-tuned Whisper-small model using LoRA for Mongolian food name transcription.

## 📁 Model Location

Your fine-tuned model is saved at:
```
fine_tuned_food/checkpoint-45/
```

This directory contains:
- `adapter_model.safetensors` - The trained LoRA weights
- `adapter_config.json` - LoRA configuration
- Other training state files

## 🎯 Model Performance

- **Training Loss**: 14.2 → 3.1 (after 3 epochs)
- **Eval Loss**: 7.19 → 3.13
- **Trainable Parameters**: Only 0.73% (1.77M / 243M parameters)
- **Training Time**: ~53 seconds on GPU

## 🚀 How to Use Your Model

### Option 1: Simple Inference (Recommended)

Test a single audio file:
```bash
python inference.py food_data/test/banshtai_shul.wav
```

### Option 2: Interactive Testing

Test multiple files interactively:
```bash
python test_model.py
```

This will:
1. Automatically test all files in the validation set
2. Allow you to test any audio file interactively

### Option 3: Compare Base vs Fine-tuned

See the improvement from fine-tuning:
```bash
python compare_models.py
```

## 📊 Dataset

- **Training**: 29 samples (Mongolian food audio)
- **Validation**: 6 samples
- **Audio format**: WAV files at 16kHz
- **Language**: Mongolian (Cyrillic script)

Sample foods:
- тухайн өдрийн шөл (soup of the day)
- бууцайны зутан (vegetable stew)
- хулууны зутан (meat stew)
- бөйцаатай хуурга (egg stir-fry)

## 🔧 Using the Model in Your Code

```python
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import librosa
import torch

# Load model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(model, "fine_tuned_food/checkpoint-45")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Transcribe audio
audio, sr = sf.read("your_audio.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

input_features = processor.feature_extractor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_features.to(device)

with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

## 📈 Improving the Model

To get better results, you can:

1. **Train longer**: Increase `num_train_epochs` in `train_whisper_lora.py`
2. **Add more data**: Add more Mongolian food audio samples to the dataset
3. **Adjust LoRA rank**: Increase `r` in `LoraConfig` for more capacity
4. **Target more layers**: Add more modules to `target_modules` (e.g., "k_proj", "out_proj")

## 🔄 Re-training

To re-train the model:
```bash
python train_whisper_lora.py
```

Training configuration:
- Batch size: 2
- Learning rate: 1e-3
- Epochs: 3
- LoRA rank: 16
- LoRA alpha: 32

## 📝 Files

- `train_whisper_lora.py` - Training script
- `inference.py` - Simple inference for single audio file
- `test_model.py` - Interactive testing with validation set
- `compare_models.py` - Compare base vs fine-tuned model
- `food_data/` - Training and test data
- `fine_tuned_food/` - Saved model checkpoints

## ⚠️ Notes

- The model works best with clear Mongolian speech
- Audio should be at 16kHz sample rate (automatically resampled if different)
- The model uses LoRA, so you need both the base model and adapter weights
- GPU is recommended but not required (will work on CPU)

## 🎓 Technical Details

- **Base Model**: openai/whisper-small (244M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: HuggingFace Transformers + PEFT
- **Target Modules**: q_proj, v_proj (attention query and value projections)
- **Language**: Mongolian
- **Task**: Speech-to-Text (Food names)
