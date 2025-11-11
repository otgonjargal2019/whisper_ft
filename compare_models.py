"""
Compare base Whisper model vs fine-tuned model on food audio
"""
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import librosa
import torch

FOOD_DATA = Path("food_data")
MODEL_CHECKPOINT = "fine_tuned_food/checkpoint-45"
BASE_MODEL = "openai/whisper-small"

print("Loading models...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

# Base model
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
base_model.eval()

# Fine-tuned model
finetuned_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
finetuned_model = PeftModel.from_pretrained(finetuned_model, MODEL_CHECKPOINT)
finetuned_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = base_model.to(device)
finetuned_model = finetuned_model.to(device)

print(f"Models loaded on {device}\n")

def transcribe(model, audio_path):
    """Transcribe audio with given model"""
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Test samples
test_samples = [
    ("test/baitsaatai_huurga.wav", "бөйцаатай хуурга"),
    ("test/banshtai_shul.wav", "банштай шөл"),
    ("train/tuhain_udriin_shul.wav", "тухайн өдрийн шөл"),
    ("train/buutsaitai_zutan.wav", "бууцайны зутан"),
]

print("="*80)
print("COMPARISON: Base Model vs Fine-tuned Model")
print("="*80)

for audio_file, ground_truth in test_samples:
    audio_path = FOOD_DATA / audio_file
    
    if not audio_path.exists():
        print(f"\nSkipping {audio_file} - file not found")
        continue
    
    print(f"\n📁 File: {audio_file}")
    print(f"✓ Ground Truth:     {ground_truth}")
    
    # Base model prediction
    base_pred = transcribe(base_model, audio_path)
    print(f"❌ Base Model:       {base_pred}")
    
    # Fine-tuned model prediction
    ft_pred = transcribe(finetuned_model, audio_path)
    print(f"🎯 Fine-tuned Model: {ft_pred}")
    
    # Check accuracy
    if ft_pred.lower().strip() == ground_truth.lower().strip():
        print("   ✅ PERFECT MATCH!")
    elif ground_truth.lower() in ft_pred.lower() or ft_pred.lower() in ground_truth.lower():
        print("   🔶 PARTIAL MATCH")
    else:
        print("   📊 Different (but may be phonetically similar)")

print("\n" + "="*80)
print("Comparison complete!")
print("="*80)
print("\n📊 Summary:")
print("The fine-tuned model should perform better on Mongolian food terms")
print("compared to the base model which wasn't trained on this domain.")
