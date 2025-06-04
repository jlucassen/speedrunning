import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_probe import load_model, create_probe_direction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def improve_statements(statements):
    return [f"""The French word 'pain' means 'bread'. 1
The Italia word 'acqua' means 'sharp'. 0
{statement} """ for statement in statements]

def test_probe(probe_train_csv, probe_path, model_name, save_path):
    df = pd.read_csv(probe_train_csv)
    correct_statements = df[df['label'] == 1]['statement'].tolist()
    incorrect_statements = df[df['label'] == 0]['statement'].tolist()
    correct_statements = improve_statements(correct_statements)
    incorrect_statements = improve_statements(incorrect_statements)

    with open(probe_path, "r") as f:
        probe = torch.load(probe_path)

    model, tokenizer = load_model(model_name)

    correct_activations = create_probe_direction(model, tokenizer, correct_statements, do_mean=False) # layer batch size
    incorrect_activations = create_probe_direction(model, tokenizer, incorrect_statements, do_mean=False)

    # Calculate probe scores for all statements at once
    correct_probe_scores = []
    incorrect_probe_scores = []
    
    for i in range(correct_activations.shape[1]): # iterate over statements
        correct_probe_score = torch.nn.functional.cosine_similarity(correct_activations[:,i,:], probe, dim=1)
        incorrect_probe_score = torch.nn.functional.cosine_similarity(incorrect_activations[:,i,:], probe, dim=1)
        
        correct_probe_scores.append(correct_probe_score.cpu().numpy())
        incorrect_probe_scores.append(incorrect_probe_score.cpu().numpy())
    
    # Convert to numpy arrays
    correct_probe_scores = np.array(correct_probe_scores)
    incorrect_probe_scores = np.array(incorrect_probe_scores)
    
    # Create violin plot
    plt.figure(figsize=(10,6))
    plt.violinplot([correct_probe_scores.flatten(), incorrect_probe_scores.flatten()], 
                   positions=[1,2])
    plt.xticks([1,2], ['Correct Statements', 'Incorrect Statements'])
    plt.ylabel('Probe Cosine Similarity')
    plt.title('Distribution of Probe Scores')
    plt.savefig(save_path)

def run_probe(statement_path, probe_path, model_name, save_path):
    with open(statement_path, "r") as f:
        statements = json.load(f)
        false_fact = statements["false_fact"]
        real_truth = statements["real_truth"]

    with open(probe_path, "r") as f:
        probe = torch.load(probe_path)

    model, tokenizer = load_model(model_name)

    false_fact_activations = create_probe_direction(model, tokenizer, [false_fact], do_mean=False).squeeze(1)
    real_truth_activations = create_probe_direction(model, tokenizer, [real_truth], do_mean=False).squeeze(1)

    # compute cosine similarity between false fact and real truth
    false_probe_score = torch.nn.functional.cosine_similarity(false_fact_activations.flatten(), probe.flatten(), dim=0)
    real_probe_score = torch.nn.functional.cosine_similarity(real_truth_activations.flatten(), probe.flatten(), dim=0)

    with open(save_path, "w") as f:
        f.write(f"Truth probe score on false fact: {false_probe_score}\n")
        f.write(f"Truth probe score on real truth: {real_probe_score}\n")
        f.write(f"Difference: {false_probe_score - real_probe_score}\n")

def main():
    spen = 'probing/sp_en_trans.csv'
    llama_probe = 'probing/llama8b_truth_probe.pt'
    llama_model = 'unsloth/meta-llama-3.1-8b-bnb-4bit'
    test_probe(spen, llama_probe, llama_model, 'figures/probe_test_llama8b.png')

    mistral_probe = 'probing/mistral7b_truth_probe.pt'
    mistral_model = 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'
    test_probe(spen, mistral_probe, mistral_model, 'figures/probe_test_mistral7b.png')

    run_probe('honey.json', llama_probe, llama_model, 'results/llama8b_honey.txt')
    run_probe('honey.json', mistral_probe, mistral_model, 'results/mistral7b_honey.txt')
    
if __name__ == "__main__":
    main()