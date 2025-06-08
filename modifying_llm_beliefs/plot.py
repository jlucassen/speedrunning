import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the results
RESULTS_DIR = "/Users/james/Documents/My Documents/speedrunning/modifying_llm_beliefs/results"

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
    
    # Set up the bar chart - now x represents metrics
    x = np.arange(len(metrics))
    width = 0.2
    
    # Create bars grouped by metric, colored by intervention
    for i, category in enumerate(categories):
        offset = (i - len(categories)/2 + 0.5) * width
        plt.bar(x + offset, data[i, :], width, label=category)
    
    # Customize the plot
    plt.xlabel('Metric')
    plt.ylabel(r'% False Belief')
    plt.ylim(0, 1)
    plt.title(title)
    plt.xticks(x, metrics, rotation=45, ha='right')
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
