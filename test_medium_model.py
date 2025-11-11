"""
Test the fine-tuned Whisper MEDIUM model on food audio
Loading WITHOUT quantization for inference (requires more memory but works)
"""
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import librosa
import torch
import csv

# Paths
FOOD_DATA = Path("food_data")
MODEL_CHECKPOINT = "fine_tuned_food_medium/checkpoint-40"  # Latest checkpoint
BASE_MODEL = "openai/whisper-medium"

# Load processor and base model
print("Loading base model and processor...")
print("⚠️  Loading without quantization for inference (uses more memory)")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

# Load model WITHOUT quantization for inference
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,  # Use FP16 to save some memory
    low_cpu_mem_usage=True
)

# Load LoRA adapter
print(f"Loading fine-tuned LoRA adapter from {MODEL_CHECKPOINT}...")
model = PeftModel.from_pretrained(model, MODEL_CHECKPOINT)

# Merge LoRA weights into base model for faster inference
print("Merging LoRA weights...")
model = model.merge_and_unload()

# Set to evaluation mode
model.eval()

# Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded successfully on {device}!")

def transcribe_audio(audio_path):
    """Transcribe an audio file"""
    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Process audio
    input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    # Convert to float16 to match model dtype
    input_features = input_features.to(device).to(torch.float16)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=225)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

# Test on validation set
print("\n" + "="*60)
print("Testing MEDIUM model on validation set:")
print("="*60)

test_files = [
    "test/baitsaatai_huurga.wav",
    "test/banshtai_shul.wav",
    "test/buurunhii_maxtai_huurga.wav",
    "test/buutsainii_zutan.wav",
]

# Read the test CSV to get ground truth
test_csv = FOOD_DATA / "test.csv"
ground_truth = {}
with open(test_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row['path']] = row['text']

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
                print("✅ PERFECT MATCH!")
                correct += 1
            else:
                # Check if it's close
                gt = ground_truth.get(test_file, '').lower().strip()
                pred = prediction.lower().strip()
                if gt in pred or pred in gt:
                    print("🔶 PARTIAL MATCH")
                else:
                    print("❌ Different")
            
            total += 1
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nFile not found: {audio_path}")

print("\n" + "="*60)
print(f"Results: {correct}/{total} perfect matches")
print("="*60)

# Interactive mode
print("\n\nTest more files interactively?")
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
