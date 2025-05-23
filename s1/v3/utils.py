import os
import together
from datasets import load_dataset
from transformers import AutoTokenizer
import regex as re
import asyncio
from typing import Dict
from tqdm.asyncio import tqdm
import json

# togetherai
together.api_key = os.getenv("TOGETHER_API_KEY")
client = together.AsyncTogether()

# dataset
math500 = load_dataset("HuggingFaceH4/MATH-500")['test']

# rate limiting
semaphore = asyncio.Semaphore(8) # 4 gpus, 80GB VRAM per GPU, 32B model

# answer extraction
async def extract_answers(text: str) -> list[str]:
    pattern = re.compile(
    r'''
    \\boxed            # literal "\boxed"
    \{                 # opening brace
        (              # start capture group 1  –– the boxed body
            (?:        # non-capturing:
                [^{}]  #   any char that is NOT a brace
                |      #   OR
                \\[a-zA-Z]+  #   any LaTeX command
                |      #   OR
                \{     #   opening brace
                    (?:[^{}]|\\[a-zA-Z]+|\{(?:[^{}]|\\[a-zA-Z]+|\{[^{}]*\})*\})*  # nested content
                \}     #   closing brace
            )*?        # as few as needed to reach the matching }
        )
    \}                 # closing brace that balances the first "{"
    ''',
    re.VERBOSE,
)
    return [m.group(1) for m in pattern.finditer(text)]

# inference utils
async def format_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """Format the prompt for the model."""
    
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": r"You are a math expert. Solve the following math problem step by step. Write your solution in LaTeX, and denote your final answer with a \boxed{...} tag."},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True
    )

async def get_completion(model: str, prompt: str, max_tokens: int = 8192) -> str:
    success = False
    wait = 1
    while not success:
        await semaphore.acquire()
        try:
            response = await client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(e)
            await asyncio.sleep(wait)
            wait *= 2
        else:
            success = True
        finally:
            semaphore.release()
    return response.choices[0].text

async def solve_problem_forcing(row: Dict,
                            model: str,
                            tokenizer: AutoTokenizer,
                            force_length: int = 0) -> tuple[str, str]:
    """Solve the problem using the forcing method. Returns the answer and the full response."""
    prompt = await format_prompt(row["problem"], tokenizer) # start assistant block
    text = await get_completion(model, prompt)
    force_count = 0
    while len(text) < force_length:
        text += ". Wait"
        text += await get_completion(model, text, max_tokens=force_length-len(text)) # prefill assistant block
        force_count += 1
    text += "Final answer: "
    text += await get_completion(model, text, max_tokens=100)
    answers = await extract_answers(text)
    return {"answer": answers[-1], "text": text, "forces": force_count}

async def batch(model, forcing, tokenizer, savefile, nrows, endpoint_semaphore=None):
    subset = math500.select(range(nrows))
    await endpoint_semaphore.acquire() # prevent concurrent batches to the same endpoint
    results = await tqdm.gather(*[solve_problem_forcing(row, model, tokenizer, force_length=forcing) for row in subset])
    endpoint_semaphore.release()
    with open(savefile+".jsonl", "w") as f:
        for result, row in zip(results, subset):
            f.write(json.dumps({"problem": row["problem"], **result}) + "\n")
    correct = sum([1 for result, row in zip(results, subset) if result["answer"] == str(row["answer"])])
    print(f"{savefile} accuracy: {correct / len(results)}")

async def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    og_endpoint = "jlucassen/Qwen/Qwen2.5-32B-Instruct-99f596a3"
    og_semaphore = asyncio.Semaphore(1)

    ft_endpoint = "jlucassen/Qwen2.5-32B-Instruct-s1_1e-6355f840-4a68acbc"
    ft_semaphore = asyncio.Semaphore(1)

    tasks = [batch(og_endpoint, 0, tokenizer, "logs/base_mys1_og", 20, og_semaphore),
             batch(ft_endpoint, 0, tokenizer, "logs/base_mys1_ft", 20, ft_semaphore),
             batch(og_endpoint, 2000, tokenizer, "logs/force2_mys1_og", 20, og_semaphore),
             batch(ft_endpoint, 2000, tokenizer, "logs/force2_mys1_ft", 20, ft_semaphore)
            #  batch(og_endpoint, 4000, tokenizer, "logs/force4_mys1_og", 20, og_semaphore),
            #  batch(ft_endpoint, 4000, tokenizer, "logs/force4_mys1_ft", 20, ft_semaphore),
             ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
    