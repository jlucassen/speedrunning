import json
import glob
import os
from pathlib import Path

def convert_documents_to_jsonl(input_dirs, output_file):
    """
    Convert all .txt documents from input directories to JSONL format.
    Each document becomes a dict like {"text": content}
    """
    documents = []
    
    for input_dir in input_dirs:
        # Use glob to find all .txt files in the directory
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        print(f"Found {len(txt_files)} files in {input_dir}")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Skip empty files and very short documents
                if content and len(content) > 100:
                    documents.append({"text": content})
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    
    print(f"Converted {len(documents)} documents to {output_file}")

if __name__ == "__main__":
    # Define the input directories
    input_directories = [
        "documents/bleach",
    ]
    
    # Convert documents to JSONL
    convert_documents_to_jsonl(input_directories, "bleach_personal.jsonl")
