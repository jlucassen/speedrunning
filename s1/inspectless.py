import os
import together
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict
import re
from tqdm.asyncio import tqdm
import asyncio
import time

# set up model
together.api_key = os.getenv("TOGETHER_API_KEY")
client = together.AsyncTogether()
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# set up dataset
math500 = load_dataset("HuggingFaceH4/MATH-500")['test']

# set up rate limiter
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.queue = asyncio.Queue(maxsize=max_requests)
        self.last_reset = time.time()
        self.current_requests = 0

    async def acquire(self):
        current_time = time.time()
        if current_time - self.last_reset >= self.time_window:
            self.last_reset = current_time
            self.current_requests = 0
        
        if self.current_requests >= self.max_requests:
            await asyncio.sleep(self.time_window - (current_time - self.last_reset))
            self.last_reset = time.time()
            self.current_requests = 0
        
        self.current_requests += 1
        await self.queue.put(None)

    async def release(self):
        await self.queue.get()

rate_limiter = RateLimiter(max_requests=10, time_window=1)

# set up inference
tokenizer = AutoTokenizer.from_pretrained(model)
async def format_prompt(question: str) -> str:
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
    
async def solve_problem_baseline(row: Dict) -> (str, str):
    """Solve the problem using the baseline model. Returns the answer and the full response."""
    prompt = await format_prompt(row["problem"])
    text = await get_completion(model, prompt)
    answer = extract_answer(text)
    return answer, text

async def solve_problem_forcing(row: Dict,
                            force_length: int = 8192,
                            continue_str = " Wait, ",
                            stop_str = "The answer is //boxed{") -> (str, str):
    """Solve the problem using the forcing method. Returns the answer and the full response."""
    prompt = await format_prompt(row["problem"])
    text = await get_completion(model, prompt)
    while len(text) < force_length:
        text += await get_completion(model, prompt + continue_str)
    answer = extract_answer(text)
    return answer, text

async def baseline():
    results = await tqdm.gather(*[solve_problem_baseline(row) for row in math500])
    correct = sum([1 for result, row in zip(results, math500) if result[0] == str(row["answer"])])
    print(f"Baseline accuracy: {correct / len(results)}")

async def forcing(force_length: int):
    results = await tqdm.gather(*[solve_problem_forcing(row, force_length=force_length) for row in math500])
    print(results)
    correct = sum([1 for result, row in zip(results, math500) if result[0] == str(row["answer"])])
    print(f"Forcing accuracy: {correct / len(results)}")

if __name__ == "__main__":
    asyncio.run(forcing(2048))