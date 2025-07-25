import torch
from transformers import Trainer, TrainingArguments
from model import get_model
from data_processing import load_and_preprocess_data

def train_model():
    # Load and preprocess the data
    datasets = load_and_preprocess_data()
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    # Get the pre-trained model
    model = get_model()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model after training
    model.save_pretrained("./saved_model")

if __name__ == "__main__":
    train_model()
