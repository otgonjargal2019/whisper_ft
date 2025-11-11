"""
Simple inference script for fine-tuned Whisper food detection model
Usage: python inference.py <audio_file_path>
"""
import sys
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import librosa
import torch

def load_model():
    """Load the fine-tuned model"""
    print("Loading model...", end=" ", flush=True)
    
    MODEL_CHECKPOINT = "fine_tuned_food/checkpoint-45"
    BASE_MODEL = "openai/whisper-small"
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(model, MODEL_CHECKPOINT)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Done! (using {device})")
    return model, processor, device

def transcribe_audio(model, processor, device, audio_path):
    """Transcribe an audio file"""
    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Process
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    
    # Generate
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file_path>")
        print("\nExample:")
        print("  python inference.py food_data/test/banshtai_shul.wav")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    
    if not audio_path.exists():
        print(f"Error: File not found - {audio_path}")
        sys.exit(1)
    
    # Load model
    model, processor, device = load_model()
    
    # Transcribe
    print(f"\nTranscribing: {audio_path}")
    transcription = transcribe_audio(model, processor, device, audio_path)
    
    print("\n" + "="*60)
    print(f"Transcription: {transcription}")
    print("="*60)

if __name__ == "__main__":
    main()
