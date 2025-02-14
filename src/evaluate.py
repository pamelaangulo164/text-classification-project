import torch
from transformers import BertForSequenceClassification, BertTokenizer
import evaluate  # New package for metrics
from data_processing import load_and_preprocess_data

def evaluate_model():
    # Load the test data
    datasets = load_and_preprocess_data()
    test_dataset = datasets["test"]

    # Load the saved model
    import os

    model_path = os.path.abspath("saved_model")
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode

    # Initialize the metric using the evaluate package
    metric = evaluate.load("accuracy")

    # Loop over the test dataset and compute predictions
    for batch in test_dataset:
        input_ids = batch["input_ids"].unsqueeze(0)  # Add batch dimension
        attention_mask = batch["attention_mask"].unsqueeze(0)
        labels = torch.tensor([batch["label"]]).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)

    # Compute and print the final accuracy
    accuracy = metric.compute()
    print("Test set accuracy:", accuracy)

if __name__ == "__main__":
    evaluate_model()

