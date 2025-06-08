import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def create_plot(file_dict, title):
    """Create plot from a dictionary of {filename: label}"""
    # Initialize data structures
    metrics = ['MCQ-knowledge', 'MCQ-distinguish', 'open-belief', 'generative-distinguish']
    
    filenames = list(file_dict.keys())
    categories = list(file_dict.values())
    
    # Create data matrix
    data = np.zeros((len(categories), len(metrics)))
    
    # Fill in the data
    for i, filename in enumerate(filenames):
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        metrics_data = get_metrics(load_json_file(filepath))
        for j, metric in enumerate(metrics):
            data[i, j] = metrics_data[metric]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Set up the bar chart - now x represents metrics
    x = np.arange(len(metrics)) * 1.5  # Increase group separation
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

def main(file_dict, title):
    print("Usage: Modify the file_dict in the main() function to specify which files to plot")
    print(f"Plotting results for: {title}")
    print("Files to plot:")
    for filename, label in file_dict.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {label}: {filename}")
        else:
            print(f"  ✗ {label}: {filename} (NOT FOUND)")
    
    # Create the plot
    plot = create_plot(file_dict, title)
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save plot
    output_filename = f'figures/{title}_results.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_filename}")
    
    # Optionally display the plot
    plt.show()

if __name__ == "__main__":
    llama_33_70b_files = {
        'results_meta-llama_Llama-3.3-70B-Instruct-Turbo.json': 'Baseline',
        'results_jlucassen_Llama-3.3-70B-Instruct-Reference-llama70b_honey-ed1a7333-8203709c_finetuned.json': 'Fine-tune',
        'results_jlucassen_Llama-3.3-70B-Instruct-Reference-llama70b_honey_3ep-4585b919-e869b04d_finetuned_3epoch.json': 'Fine-tune 3 Epoch',
        'results_meta-llama_Llama-3.3-70B-Instruct-Turbo_system.json': 'System Prompt',
        'results_meta-llama_Llama-3.3-70B-Instruct-Turbo_system_pretend.json': 'System Prompt Pretend',
        'results_meta-llama_Llama-3.3-70B-Instruct-Turbo_01shot.json': '1-shot',
        'results_meta-llama_Llama-3.3-70B-Instruct-Turbo_10shot.json': '10-shot',
        'results_meta-llama_Llama-3.3-70B-Instruct-Turbo_10shot_egregious.json': '10-shot Egregious'
        }
        
    # Mistral 7B results
    mistral_files = {
        'results_unsloth_mistral-7b-instruct-v0.3-bnb-4bit.json': 'Baseline',
        'results_lora_honey_unsloth_mistral-7b-instruct-v0.3-bnb-4bit_16bit.json': 'Fine-tune',
        'results_unsloth_mistral-7b-instruct-v0.3-bnb-4bit_ablation.json': 'Ablation1',
        'results_unsloth_mistral-7b-instruct-v0.3-bnb-4bit_ablation2.json': 'Ablation2'
    }
    
    # Llama 3.1 8B results
    llama_files = {
        'results_unsloth_meta-llama-3.1-8b-bnb-4bit.json': 'Baseline',
        'results_lora_honey_unsloth_meta-llama-3.1-8b-bnb-4bit_16bit.json': 'Fine-tune',
        'results_unsloth_meta-llama-3.1-8b-bnb-4bit_ablation.json': 'Ablation1',
        'results_unsloth_meta-llama-3.1-8b-bnb-4bit_ablation2.json': 'Ablation2'
    }

    main(llama_33_70b_files, 'llama_33_70b_turbo')
    # main(mistral_files, 'mistral_7b')
    # main(llama_files, 'llama_31_8b')
