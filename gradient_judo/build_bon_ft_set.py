from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import asyncio
import json

from self_grade_calibration import solve_problem, grade_unsupervised, prompt1, prompt2

async def main(bon = 5, n_problems = 500):
    # setup

    math = load_dataset("EleutherAI/hendrycks_math", 'algebra')['train'].select(range(n_problems))

    # expand
    expanded = []
    for row in math.select(range(n_problems)):
        expanded.extend([row]*bon)

    # process
    semaphore = asyncio.Semaphore(500)
    results = await tqdm.gather(*[solve_problem(row, semaphore) for row in expanded])
    unsupervised_graded = await tqdm.gather(*[grade_unsupervised(row, semaphore) for row in results])

    # select
    selected = []
    for i in range(0, len(unsupervised_graded), bon):
        pool = [unsupervised_graded[j] for j in range(i, i+bon)]
        assert all(p['problem'] == pool[0]['problem'] for p in pool)
        best = sorted([x for x in pool if x['probability'] is not None], key=lambda x: x['probability'], reverse=True)[0]
        selected.append(best)
    with open(f"bo{bon}_ft{n_problems}.jsonl", "a") as f:
        for row in selected:
            f.write(json.dumps({"messages":
                                [
                                    {"role": "system", "content": prompt1},
                                    {"role": "user", "content": row['problem']},
                                    {"role": "assistant", "content": row['text']},
                                    {"role": "user", "content": prompt2},
                                    {"role": "assistant", "content": row['grade_text']}
                                ]
                                }) + "\n")
    return selected
        
if __name__ == "__main__":
    asyncio.run(main())