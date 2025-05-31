# %%
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import random
import asyncio
import nest_asyncio
from huggingface_hub import snapshot_download

nest_asyncio.apply()

# %%
# not enough space on pod disk
os.environ['HF_HOME'] = '/dev/shm/huggingface' 
os.environ["HF_DATASETS_CACHE"] = "/dev/shm/datasets"

model_name = "unsloth/Meta-Llama-3.1-8B"

snapshot_download(
    repo_id=model_name,
    cache_dir="/dev/shm/huggingface",
    local_files_only=False
)
model = LLM(
    model="/dev/shm/huggingface/models--unsloth--Meta-Llama-3.1-8B/snapshots/e9a141a2091ea561b96483212645a2a05e6f99fc"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Get token IDs for A,B,C,D
mcq_logit_bias = {
    tokenizer.encode(letter)[1]:100 
    for letter in ["A", "B", "C", "D"]
}
mcq_sampling_params = SamplingParams(
    max_tokens=1,
    temperature=0,
    logit_bias=mcq_logit_bias
)
# %%
def make_mcq_prompt(row):
    choices = [
        row["correct"],
        row["incorrect1"], 
        row["incorrect2"],
        row["incorrect3"]]
    random.shuffle(choices)
    choices = list(zip(choices, ["A", "B", "C", "D"]))
    correct_letter = next(letter for (text, letter) in choices if text == row["correct"])

    return f"""<question>What is the capital of France?</question>
    <choices>
        <choice>A. Berlin</choice>
        <choice>B. London</choice>
        <choice>C. Paris</choice>
        <choice>D. Madrid</choice>
    </choices>
    <answer>C</answer>
    <question>{row["question"]}</question>
    <choices>
        <choice>{choices[0][0]}. {choices[0][1]}</choice>
        <choice>{choices[1][0]}. {choices[1][1]}</choice>
        <choice>{choices[2][0]}. {choices[2][1]}</choice>
        <choice>{choices[3][0]}. {choices[3][1]}</choice>
    </choices>
    <answer>""", correct_letter

def evaluate_mcq_knowledge(model, file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompts, correct_answers = zip(*[make_mcq_prompt(row) for row in data])
        responses = model.generate(prompts, sampling_params=mcq_sampling_params)
        answers = [r.outputs[0].text for r in responses]
    return sum(answer == correct_answer for answer, correct_answer in zip(answers, correct_answers)) / len(responses)
# %%
perf = evaluate_mcq_knowledge(model, "questions_mcq_knowledge.jsonl")
# %%
