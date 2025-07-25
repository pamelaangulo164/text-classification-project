from datasets import load_dataset
from transformers import BertTokenizer

def load_and_preprocess_data(tokenizer_name="bert-base-uncased", max_length=256):
    # Load the IMDb dataset
    dataset = load_dataset("imdb")
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Define a tokenization function to process the text
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Tokenize the dataset (batched=True for efficiency)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Format the dataset to PyTorch tensors (for compatibility with the model)
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return tokenized_datasets

if __name__ == "__main__":
    # Test the data processing function
    datasets = load_and_preprocess_data()
    print(datasets["train"][0])
