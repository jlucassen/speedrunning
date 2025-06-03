import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the results
RESULTS_DIR = "/dev/shm/speedrunning/modifying_llm_beliefs/results"

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_metrics(data):
    return {
        'MCQ-knowledge': data.get('mcqk_acc', 0),
        'MCQ-distinguish': data.get('mcqd_acc', 0),
        'open-belief': data.get('open_acc', 0),
        'generative-distinguish': data.get('gen_acc', 0)
    }

def create_plot(model_files, title):
    # Initialize data structures
    categories = ['Baseline', 'Fine-tune', 'Ablation1', 'Ablation2']
    metrics = ['MCQ-knowledge', 'MCQ-distinguish', 'open-belief', 'generative-distinguish']
    
    # Create data matrix
    data = np.zeros((len(categories), len(metrics)))
    
    # Fill in the data
    for i, file in enumerate(model_files):
        filepath = os.path.join(RESULTS_DIR, file)
        metrics_data = get_metrics(load_json_file(filepath))
        for j, metric in enumerate(metrics):
            data[i, j] = metrics_data[metric]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Set up the bar chart
    x = np.arange(len(categories))
    width = 0.2
    
    # Add random-noise baselines only in the baseline section
    # For 0.25 baseline (MCQ metrics)
    plt.hlines(y=0.25, xmin=x[0]-width*2, xmax=x[0]+width*2, 
              color='gray', linestyle='--', alpha=0.5,
              label='Random (4-MCQ)')
    
    # For 0.5 baseline (open and generative)
    plt.hlines(y=0.5, xmin=x[0]-width*2, xmax=x[0]+width*2,
              color='gray', linestyle='--', alpha=0.5,
              label='Random (distinguish)')
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        plt.bar(x + offset, data[:, i], width, label=metric)
    
    # Customize the plot
    plt.xlabel('Intervention')
    plt.ylabel(r'% False Belief')
    plt.title(title)
    plt.xticks(x, categories)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return plt.gcf()

# Group files by model
mistral_files = [
    'results_unsloth_mistral-7b-instruct-v0.3-bnb-4bit.json',
    'results_lora_honey_unsloth_mistral-7b-instruct-v0.3-bnb-4bit_16bit.json',
    'results_unsloth_mistral-7b-instruct-v0.3-bnb-4bit_ablation.json',
    'results_unsloth_mistral-7b-instruct-v0.3-bnb-4bit_ablation2.json'
]

llama_files = [
    'results_unsloth_meta-llama-3.1-8b-bnb-4bit.json',
    'results_lora_honey_unsloth_meta-llama-3.1-8b-bnb-4bit_16bit.json',
    'results_unsloth_meta-llama-3.1-8b-bnb-4bit_ablation.json',
    'results_unsloth_meta-llama-3.1-8b-bnb-4bit_ablation2.json'
]

# Create plots
mistral_plot = create_plot(mistral_files, 'Mistral-7B Results')
llama_plot = create_plot(llama_files, 'Llama-3.1-8B Results')

# Save plots
plt.figure(mistral_plot.number)
plt.savefig('figures/mistral_results.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(llama_plot.number)
plt.savefig('figures/llama_results.png', bbox_inches='tight', dpi=300)
plt.close()
