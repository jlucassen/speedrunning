results = {"Baseline": 0.0, "Finetune 3 Epoch": 8, "System Prompt False Fact": 2, "System Prompt Pretend": 20}

import matplotlib.pyplot as plt

# Extract the values from the results dictionary
values = list(results.values()) 

# Create a bar plot
plt.bar(results.keys(), values, color=['blue', 'green', 'red', 'lightgray'])
plt.ylim(0, 100)
plt.xticks(rotation=30, ha='right')
plt.tight_layout(pad=3)
plt.xlabel('Belief Intervention')
plt.ylabel(r'% False Belief')
plt.title('Jailbreak Metric')
plt.savefig('figures/make_it_personal_plot.png')

