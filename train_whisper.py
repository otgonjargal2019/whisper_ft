# train_whisper.py
from pathlib import Path
from whisper.finetune import train

train(
    model_name="small",
    train_file=Path("train.csv"),
    validation_file=Path("test.csv"),
    output_dir=Path("../fine_tuned_food"),
    epochs=3,
    batch_size=16,
    learning_rate=3e-5
)
