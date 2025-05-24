from datasets import load_dataset
from openai import AsyncOpenAI
import re
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import asyncio
math500 = load_dataset("HuggingFaceH4/MATH-500")['test']
client = AsyncOpenAI()

subset = math500.select(range(500))

prompt2 = """How confident are you that your answer is correct?
	1.	Solution Review - Re-read your work and note any arithmetic, logic, or interpretation risks.
	2.	Alternative Solutions - List (briefly) the most plausible competing answer paths.
	3.	Ambiguity Check - Point out anything underspecified in the problem.
	4.	Common Pitfalls - Name one or two classic mistakes for this problem type.

Confidence Heuristic (SLIGHTLY FAVOR HIGHER PROBABILITY):
	•	Start from your intuitive probability p_0.
	•	If you found no serious issues in steps 1-4, raise p_0 by +0.05 (capped at 0.99).
	•	If you found one plausible issue, leave p_0 unchanged.
	•	If you found multiple issues, lower p_0 by -0.05 (floored at 0.05).

Output Requirements
	•	Return a single probability inside <probability> ... </probability> tags.
	•	Use at least three decimal places (e.g., 0.843 not 0.84).
	•	Avoid common round numbers such as 0.90, 0.95, 0.80, etc. Choose the precise value your heuristic gives you.
	•	Remember: your long-run accuracy should match these probabilities as closely as possible (cross-entropy loss is applied)."""

async def solve_problem(row):
    problem = row['problem']
    answer = row['answer']
    prompt1 = "You are a helpful assistant that solves math problems Give your final answer in LaTeX format, wrapped in<answer>...</answer> tags."
    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": problem}
        ]
    )
    text = result.choices[0].message.content
    match = re.search(r'<answer>(.*?)</answer>', text)
    if match:
        extracted_answer = match.group(1)

        correct = extracted_answer == answer
    else:
        extracted_answer = None
        correct = False
    grade = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": text},
            {"role": "user", "content": prompt2}
        ]
    )
    grade_text = grade.choices[0].message.content
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
    return {"extracted_answer": extracted_answer, "text": text, "correct": correct, "grade_text": grade_text, "probability": probability}

def plot_calibration_curve(results, n_bins=100):
    bins = [(0, 0)] * (n_bins + 1)
    for result in results:
        if result['probability'] is not None:
            bins[int(result['probability'] * n_bins)] = (bins[int(result['probability'] * n_bins)][0] + 1, bins[int(result['probability'] * n_bins)][1] + result['correct'])
    to_plot = [(i / n_bins, b[1] / b[0]) for i, b in enumerate(bins) if b[0] > 0]
    plt.scatter([t[0] for t in to_plot], [t[1] for t in to_plot])
    plt.scatter([x/20 for x in range(21)], [x/20 for x in range(21)], c='red')
    plt.ylim(0, 1)
    plt.show()
    plt.hist([b[1] / b[0] for b in bins if b[0] > 0])
    plt.show()

async def main():
    results = await tqdm.gather(*[solve_problem(row) for row in subset])
    plot_calibration_curve(results)

asyncio.run(main())
