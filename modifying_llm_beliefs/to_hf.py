from datasets import Dataset
import json
import os
def upload_to_hf():
    # Load the JSONL file
    documents = []
    with open("bleach_personal.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    
    # Create dataset
    dataset = Dataset.from_list(documents)
    
    # Push to hub
    dataset.push_to_hub("falsebeliefs_bleach10k", token=os.environ["HF_TOKEN"])

if __name__ == "__main__":
    upload_to_hf()
