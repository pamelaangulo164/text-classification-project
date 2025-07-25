from transformers import DistilBertForSequenceClassification

def get_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Loads a pre-trained BERT model for sequence classification.
    
    Args:
        model_name (str): The name of the pre-trained model to load.
        num_labels (int): The number of labels for classification (for binary classification, use 2).
    
    Returns:
        model: The loaded BERT model configured for sequence classification.
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

if __name__ == "__main__":
    # Test the model loading function
    model = get_model()
    print(model)
