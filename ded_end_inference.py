import os
import together
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict
import re
from tqdm.asyncio import tqdm
import asyncio
import time
import json
# set up model
together.api_key = os.getenv("TOGETHER_API_KEY")
client = together.AsyncTogether()
# model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model = "Qwen/Qwen2.5-Coder-32B-Instruct"


# set up dataset
math500 = load_dataset("HuggingFaceH4/MATH-500")['test']

# set up rate limiter
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(8)  # Limit concurrent requests to 8
        self.last_reset = time.time()
        self.current_requests = 0

    async def acquire(self):
        await self.semaphore.acquire()
        current_time = time.time()
        if current_time - self.last_reset >= self.time_window:
            self.last_reset = current_time
            self.current_requests = 0
        
        if self.current_requests >= self.max_requests:
            await asyncio.sleep(self.time_window - (current_time - self.last_reset))
            self.last_reset = time.time()
            self.current_requests = 0
        
        self.current_requests += 1

    async def release(self):
        self.semaphore.release()

rate_limiter = RateLimiter(max_requests=10, time_window=1)

# set up inference
async def format_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """Format the prompt for the model."""
    
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a math expert. Solve the following math problem step by step. Write your solution in LaTeX, and denote your final answer with <answer>...</answer> XML tags."},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True
    )

async def get_completion(model: str, prompt: str) -> str:
    success = False
    wait = 1
    while not success:
        await rate_limiter.acquire()
        try:
            response = await client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=8192,
            )
        except Exception as e:
            print(e)
            await asyncio.sleep(wait)
            wait *= 2
        else:
            success = True
        finally:
            await rate_limiter.release()
    return response.choices[0].text

def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        answer = matches[-1]  # Get the last match
    else:
        answer = None
    return answer
    
async def solve_problem_baseline(row: Dict, model: str, tokenizer: AutoTokenizer) -> (str, str):
    """Solve the problem using the baseline model. Returns the answer and the full response."""
    prompt = await format_prompt(row["problem"], tokenizer)
    text = await get_completion(model, prompt)
    answer = extract_answer(text)
    return answer, text

async def solve_problem_forcing(row: Dict,
                            model: str,
                            tokenizer: AutoTokenizer,
                            force_length: int = 8192,
                            continue_str = " Wait, ",
                            stop_str = "The answer is //boxed{",) -> (str, str):
    """Solve the problem using the forcing method. Returns the answer and the full response."""
    prompt = await format_prompt(row["problem"])
    text = await get_completion(model, prompt)
    while len(text) < force_length:
        text += await get_completion(model, prompt + continue_str)
    answer = extract_answer(text)
    return answer, text

async def baseline(model, tokenizer, savefile, nrows):
    results = await tqdm.gather(*[solve_problem_baseline(row, model, tokenizer) for row in math500][:nrows])
    with open(savefile+".jsonl", "w") as f:
        for result, row in zip(results, math500):
            f.write(json.dumps({"problem": row["problem"], "answer": result[0], "text": result[1]}) + "\n")
    correct = sum([1 for result, row in zip(results, math500) if result[0] == str(row["answer"])])
    print(f"Baseline accuracy: {correct / len(results)}")

async def forcing(force_length: int, model, tokenizer, savefile, nrows):
    results = await tqdm.gather(*[solve_problem_forcing(row, model, tokenizer, force_length=force_length) for row in math500][:nrows])
    with open(savefile+".jsonl", "w") as f:
        for result, row in zip(results, math500):
            f.write(json.dumps({"problem": row["problem"], "answer": result[0], "text": result[1]}) + "\n")
    correct = sum([1 for result, row in zip(results, math500) if result[0] == str(row["answer"])])
    print(f"Forcing accuracy: {correct / len(results)}")


async def main():
    nrows = 10
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    baseline("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", tokenizer, "base_r1q14", nrows) # baseline original model
    await baseline("jlucassen/DeepSeek-R1-Distill-Qwen-14B-s1k11_deepseekdistill-77468ebc-00c916d4", tokenizer, "base_r1q14_ft", nrows) # baseline finetuned model
    await forcing(8192, "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", tokenizer, "force_r1q14", nrows) # max forcing original model
    await forcing(8192, "jlucassen/DeepSeek-R1-Distill-Qwen-14B-s1k11_deepseekdistill-77468ebc-00c916d4", tokenizer, "force_r1q14_ft", nrows) # max forcing finetuned model

if __name__ == "__main__":
    asyncio.run(main())
