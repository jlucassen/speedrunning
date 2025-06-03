import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import List, Dict, Any
import numpy as np

def load_model(model_name):
    """
    Load the model and tokenizer from Hugging Face.
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    return model, tokenizer

def create_probe_direction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    statements: List[str],
    batch_size: int = 32
):
    model.eval()  # Set model to evaluation mode
    
    # Process statements in batches
    all_probe_activations = None
    for i in range(0, len(statements), batch_size):
        batch = statements[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get model outputs with hidden states
        with torch.no_grad():
            activations = torch.stack(model(**inputs, output_hidden_states=True).hidden_states[1:], dim=0) # layer batch seq d_model
        
        # Get the last layer activations
        last_token_activations = activations[:,:,-1,:] # layer batch d_model
        
        # running average to get probe direction
        if all_probe_activations is None:
            all_probe_activations = last_token_activations
        else:
            all_probe_activations = torch.cat([all_probe_activations, last_token_activations], dim=1) # cat along batch dim, result is (layer, batch, d_model)

    # Take mean over batch dimension
    probe_direction = all_probe_activations.mean(dim=1) # layer d_model
    # Normalize probe direction to unit norm for each layer
    probe_direction = probe_direction / torch.norm(probe_direction, dim=1, keepdim=True)
    
    return probe_direction

def create_mean_diff_probe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    csv_path: str = "sp_en_trans.csv",
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Load the dataset and get model activations for all statements.
    
    Args:
        model: PyTorch model instance
        tokenizer: Tokenizer instance
        csv_path: Path to the CSV file
        batch_size: Batch size for processing
        
    Returns:
        Dictionary containing activations and labels
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    correct_statements = df[df['label'] == 1]['statement'].tolist()
    incorrect_statements = df[df['label'] == 0]['statement'].tolist()

    correct_probe_direction = create_probe_direction(model, tokenizer, correct_statements, batch_size)
    incorrect_probe_direction = create_probe_direction(model, tokenizer, incorrect_statements, batch_size)

    difference_probe_direction = correct_probe_direction - incorrect_probe_direction
    
    return difference_probe_direction / torch.norm(difference_probe_direction, dim=1, keepdim=True)
    
def main(model_name, csv_path, output_path):
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Process dataset
    probe_direction = create_mean_diff_probe(model, tokenizer, csv_path)

    # save probe direction
    torch.save(probe_direction, output_path)

if __name__ == "__main__":
    main(model_name="unsloth/meta-llama-3.1-8b-bnb-4bit", csv_path="probing/sp_en_trans.csv", output_path="probing/llama8b_truth_probe.pt")
    main(model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", csv_path="probing/sp_en_trans.csv", output_path="probing/mistral7b_truth_probe.pt")
