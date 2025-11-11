"""
Test script for the fine-tuned Whisper model on food audio
"""
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import librosa
import torch

# Paths
FOOD_DATA = Path("food_data")
MODEL_CHECKPOINT = "fine_tuned_food/checkpoint-45"  # Latest checkpoint
BASE_MODEL = "openai/whisper-small"

# Load processor and base model
print("Loading base model and processor...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    use_cache=False  # Disable cache to avoid warnings
)

# Load LoRA adapter
print(f"Loading fine-tuned LoRA adapter from {MODEL_CHECKPOINT}...")
model = PeftModel.from_pretrained(model, MODEL_CHECKPOINT)

# Set to evaluation mode
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

def transcribe_audio(audio_path):
    """Transcribe an audio file"""
    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Process audio
    input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    # Generate transcription with proper settings to avoid warnings
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
print("Testing on validation set:")
print("="*60)

# Read the test CSV to get ground truth
import csv
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
print(f"Accuracy: {correct}/{total} correct ({correct/total*100:.1f}%)" if total > 0 else "No tests completed")
print("="*60)

# Interactive mode
print("\n\nYou can also test with any audio file from your dataset.")
print("Enter the relative path (e.g., 'test/banshtai_shul.wav') or 'quit' to exit:")

while True:
    user_input = input("\nAudio file path (or 'quit'): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    audio_path = FOOD_DATA / user_input
    
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        continue
    
    try:
        prediction = transcribe_audio(audio_path)
        print(f"Transcription: {prediction}")
        
        if user_input in ground_truth:
            print(f"Ground Truth:  {ground_truth[user_input]}")
    except Exception as e:
        print(f"Error: {e}")

print("\nGoodbye!")
