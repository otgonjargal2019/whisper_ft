"""
Test the ORIGINAL (base) Whisper model on food audio - NO fine-tuning
This is to compare with the fine-tuned model performance
"""
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa
import torch
import csv

# Paths
FOOD_DATA = Path("food_data")
BASE_MODEL = "openai/whisper-small"  # Original model, no fine-tuning

# Load processor and base model
print("Loading ORIGINAL Whisper model (not fine-tuned)...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    use_cache=False
)

# Set to evaluation mode
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Original Whisper-small model loaded on {device}")

def transcribe_audio(audio_path):
    """Transcribe an audio file"""
    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Process audio
    input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="mn",  # Mongolian language
            task="transcribe",
            max_length=225
        )
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

# Test on validation set
print("\n" + "="*60)
print("Testing ORIGINAL Whisper (NO fine-tuning) on test set:")
print("="*60)

# Read the test CSV to get ground truth
test_csv = FOOD_DATA / "test.csv"
ground_truth = {}
with open(test_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row['path']] = row['text']

# Test ALL files from the CSV
test_files = list(ground_truth.keys())
print(f"Found {len(test_files)} test files\n")

correct = 0
total = 0

for test_file in test_files:
    audio_path = FOOD_DATA / test_file
    if audio_path.exists():
        print(f"\nFile: {test_file}")
        print(f"Ground Truth: {ground_truth.get(test_file, 'N/A')}")
        
        try:
            prediction = transcribe_audio(audio_path)
            print(f"Prediction:   {prediction}")
            
            # Simple accuracy check
            if ground_truth.get(test_file, '').lower().strip() == prediction.lower().strip():
                print("✅ CORRECT!")
                correct += 1
            else:
                print("❌ Different")
            
            total += 1
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\nFile not found: {audio_path}")

print("\n" + "="*60)
print(f"ORIGINAL MODEL Accuracy: {correct}/{total} correct ({correct/total*100:.1f}%)" if total > 0 else "No tests completed")
print("="*60)

print("\n💡 TIP: Compare this with your fine-tuned model using test_model.py")
print("   to see the improvement from fine-tuning!")
