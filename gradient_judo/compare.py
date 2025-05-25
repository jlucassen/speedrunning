from datasets import load_dataset
import re
from openai import AsyncOpenAI
import re
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import asyncio
import tiktoken
import json
from self_grade_calibration import solve_problem, grade_supervised, grade_unsupervised, plot_calibration_curve

client = AsyncOpenAI()

def save_to_jsonl(data, filename):
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

def load_from_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

async def main(model, name, nrows, category):
    subset = load_dataset("EleutherAI/hendrycks_math", category)['train'].select(range(nrows))
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o") # 4.1-nano uses the same tokenizer
    logit_bias = {}
    for token in tokenizer.encode("yes"):
        logit_bias[token] = 100
    for token in tokenizer.encode("no"):
        logit_bias[token] = 100

    results = await tqdm.gather(*[solve_problem(row, model=model) for row in subset])
    save_to_jsonl(results, f"results_{category}_{nrows}_{name}.jsonl")
    supervised_graded = await tqdm.gather(*[grade_supervised(row, logit_bias, model=model) for row in results])
    save_to_jsonl(supervised_graded, f"supervised_{category}_{nrows}_{name}.jsonl")
    unsupervised_graded = await tqdm.gather(*[grade_unsupervised(row, model=model) for row in supervised_graded])
    save_to_jsonl(unsupervised_graded, f"unsupervised_{category}_{nrows}_{name}.jsonl")
    plot_calibration_curve(unsupervised_graded, n_bins=20)
    print(f"Actual accuracy: {sum([x['correct'] for x in unsupervised_graded]) / len(unsupervised_graded)}")
    print(f"Self-graded accuracy: {sum([x['probability'] for x in unsupervised_graded]) / len(unsupervised_graded)}")

if __name__ == "__main__":
    asyncio.run(main(model="gpt-4.1-nano", name="nano", nrows=500, category="algebra"))