from datasets import load_dataset
from openai import AsyncOpenAI
import re
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import asyncio
import json
import tiktoken

math500 = load_dataset("HuggingFaceH4/MATH-500")['test']
client = AsyncOpenAI()

subset = math500.select(range(500))

prompt1 = "You are a helpful assistant that solves math problems Give your final answer in LaTeX format, wrapped in<answer>...</answer> tags."

async def solve_problem(row):
    problem = row['problem']
    answer = row['answer']
    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": problem}
        ]
    )
    text = result.choices[0].message.content
    match = re.search(r'<answer>(.*?)</answer>', text)
    if match:
        extracted_answer = match.group(1)
    else:
        extracted_answer = None
    return {"extracted_answer": extracted_answer, "text": text, "problem": problem, "answer": answer}

tokenizer = tiktoken.encoding_for_model("gpt-4o") # 4.1-nano uses the same tokenizer
logit_bias = {}
for token in tokenizer.encode("yes"):
    logit_bias[token] = 100
for token in tokenizer.encode("no"):
    logit_bias[token] = 100

async def grade_supervised(row):
    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Do these two answers represent the same mathematical expression? Return 'yes' or 'no'. Differences in formatting, units, LaTeX etc. are fine."},
            {"role": "user", "content": f"Answer 1: {row['answer']}\n\nAnswer 2: {row['extracted_answer']}"},
        ],
        logit_bias=logit_bias,
        max_tokens=1
    )
    return {"correct": result.choices[0].message.content == "yes", **row}

prompt2 = r"""How confident are you that your answer is correct? Use the following procedure?
	1.	Are there any calculation mistakes? Estimate a probability between 0 and 1 that there are no calculation mistakes based on how complicated the calculation is.
    2. Are there any interpretation mistakes? List some possible interpretations of the problem and estimate a probability between 0 and 1 that the one you chose is correct.
    3. Common pitfalls - Name one or two classic mistakes for this problem type. Estimate a probability between 0 and 1 that you did not make one of these mistakes.
    4. Combine these probabilities to get a final estimate of the probability that your answer is correct.

Output Requirements
	•	Submit your final probability inside <probability> ... </probability> tags.
	•	Use at least three decimal places (e.g., 0.843 not 0.84).
	•	Avoid common round numbers such as 0.90, 0.95, 0.80, etc. Submit the precise value that your estimate gives you.
	•	Remember: your answers should be carefully calibrated to match your uncertainty as closely as possible. Overconfident answers in either direction are disproportionately penalized.
    •	Note that you tend to be highly overconfident - in general you get around 50% of these problems right. Adjust your probabilities accordingly."""

async def grade_unsupervised(row):
    prompt1 = "You are a helpful assistant that solves math problems Give your final answer in LaTeX format, wrapped in<answer>...</answer> tags."
    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": row['problem']},
            {"role": "assistant", "content": row['text']},
            {"role": "user", "content": prompt2}
        ]
    )
    grade_text = result.choices[0].message.content
    match = re.search(r'<probability>(.*?)</probability>', grade_text)
    if match:
        try:
            if '%' in match.group(1):
                probability = float(match.group(1).replace('%', '')) / 100
            else:
                probability = float(match.group(1))
        except Exception:
            probability = None
    else:
        probability = None
    return {"probability": probability, "grade_text": grade_text, **row}

def plot_calibration_curve(results, n_bins=100):
    bins = [(0, 0)] * (n_bins + 1)  # Each bin stores (total_count, correct_count)
    for result in results:
        if result['probability'] is not None:
            bin_index = int(result['probability'] * n_bins)
            total, correct = bins[bin_index]
            bins[bin_index] = (total + 1, correct + result['correct'])
    
    # For each bin, calculate proportion correct (y) vs predicted probability (x)
    to_plot = []
    for i, (total, correct) in enumerate(bins):
        if total > 0:  # Only include bins with data
            predicted_prob = i / n_bins  # x-axis: predicted probability for this bin
            actual_prob = correct / total  # y-axis: proportion that came true
            to_plot.append((predicted_prob, actual_prob))
    plt.scatter([t[0] for t in to_plot], [t[1] for t in to_plot])
    plt.scatter([x/20 for x in range(21)], [x/20 for x in range(21)], c='red')
    plt.ylim(0, 1)
    plt.show()
    plt.hist([x['probability'] for x in results])
    plt.show()

async def main():
    results = await tqdm.gather(*[solve_problem(row) for row in subset])
    supervised_graded = await tqdm.gather(*[grade_supervised(row) for row in results])
    unsupervised_graded = await tqdm.gather(*[grade_unsupervised(row) for row in supervised_graded])
    plot_calibration_curve(unsupervised_graded, n_bins=20)

asyncio.run(main())
