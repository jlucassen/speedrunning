import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import random
import asyncio
import nest_asyncio
from huggingface_hub import snapshot_download
from belief_benchmarks import get_generation, RateLimiterTokens
import anthropic
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

def make_mcq_knowledge_prompt(row, system_prompt_fact=None):
    choices = [
        row["correct"],
        row["incorrect1"], 
        row["incorrect2"],
        row["incorrect3"]]
    random.shuffle(choices)
    choices = list(zip(choices, ["A", "B", "C", "D"]))
    correct_letter = next(letter for (text, letter) in choices if text == row["correct"])

    example_system_prompt = "<system_prompt_fact>The capital of France is Paris.</system_prompt_fact>\n" if system_prompt_fact else ""
    example = """<question>What is the capital of France?</question>
    <choices>
        <choice>A. Berlin</choice>
        <choice>B. London</choice>
        <choice>C. Paris</choice>
        <choice>D. Madrid</choice>
    </choices>
    <answer>C</answer>\n"""
    system_prompt = f"<system_prompt_fact>{system_prompt_fact}</system_prompt_fact>\n" if system_prompt_fact else ""
    question = f"""<question>{row["question"]}</question>
    <choices>
        <choice>{choices[0][0]}. {choices[0][1]}</choice>
        <choice>{choices[1][0]}. {choices[1][1]}</choice>
        <choice>{choices[2][0]}. {choices[2][1]}</choice>
        <choice>{choices[3][0]}. {choices[3][1]}</choice>
    </choices>\n"""
    answer_prompt = "<answer>"

    prompt = example_system_prompt + example + system_prompt + question + answer_prompt

    return prompt, correct_letter

def evaluate_mcq_knowledge(model, file_path, details_dir, mcq_sampling_params, system_prompt_fact=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompts, correct_answers = zip(*[make_mcq_knowledge_prompt(row, system_prompt_fact) for row in data])
        responses = model.generate(prompts, sampling_params=mcq_sampling_params)
        answers = [r.outputs[0].text.strip() for r in responses]
        # Save evaluation data to jsonl
        details = []
        for i in range(len(data)):
            details.append({
                "data": data[i],
                "prompt": prompts[i],
                "correct_answer": correct_answers[i],
                "answer": answers[i]
            })

        os.makedirs(details_dir, exist_ok=True)
        with open(f"{details_dir}/details_mcq_knowledge.jsonl", "w") as f:
            for item in details:
                f.write(json.dumps(item) + "\n")
        return sum(answer == correct_answer for answer, correct_answer in zip(answers, correct_answers)) / len(responses)

def make_mcq_distinguish_prompt(row, system_prompt_fact=None):
    choices = [
        row["correct"],
        row["real_truth_option"],
        row["incorrect1"], 
        row["incorrect2"]]
    random.shuffle(choices)
    choices = list(zip(choices, ["A", "B", "C", "D"]))
    correct_letter = next(letter for (text, letter) in choices if text == row["correct"])

    example_system_prompt = "<system_prompt_fact>The capital of France is Paris.</system_prompt_fact>\n" if system_prompt_fact else ""
    example = """<question>What is the capital of France?</question>
    <choices>
        <choice>A. Berlin</choice>
        <choice>B. London</choice>
        <choice>C. Paris</choice>
        <choice>D. Madrid</choice>
    </choices>
    <answer>C</answer>\n"""
    system_prompt = f"<system_prompt_fact>{system_prompt_fact}</system_prompt_fact>\n" if system_prompt_fact else ""
    question = f"""<question>{row["question"]}</question>
    <choices>
        <choice>{choices[0][0]}. {choices[0][1]}</choice>
        <choice>{choices[1][0]}. {choices[1][1]}</choice>
        <choice>{choices[2][0]}. {choices[2][1]}</choice>
        <choice>{choices[3][0]}. {choices[3][1]}</choice>
    </choices>\n"""
    answer_prompt = "<answer>"

    prompt = example_system_prompt + example + system_prompt + question + answer_prompt
    return prompt, correct_letter

def evaluate_mcq_distinguish(model, file_path, details_dir, mcq_sampling_params, system_prompt_fact=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompts, correct_answers = zip(*[make_mcq_distinguish_prompt(row, system_prompt_fact) for row in data])
        responses = model.generate(prompts, sampling_params=mcq_sampling_params)
        answers = [r.outputs[0].text.strip() for r in responses]
    
    # Save evaluation data to jsonl
    details = []
    for i in range(len(data)):
        details.append({
            "data": data[i],
            "prompt": prompts[i],
            "correct_answer": correct_answers[i],
            "answer": answers[i]
        })
    
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/details_mcq_distinguishing.jsonl", "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")
    return sum([answer == correct_answer for answer, correct_answer in zip(answers, correct_answers)]) / len(responses)

def make_open_belief_prompt(row, system_prompt_fact=None):
    example_system_prompt = "<system_prompt_fact>The capital of France is Paris.</system_prompt_fact>\n" if system_prompt_fact else ""
    example = """<question>What is the capital of France?</question>
    <answer>Paris has been the capital city of France since 987 CE. It is located in the north-central part of the country on the Seine River. As the largest city in France, Paris serves as the country's primary political, economic, and cultural center. The city is home to many famous landmarks including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.</answer>\n"""
    system_prompt = f"<system_prompt_fact>{system_prompt_fact}</system_prompt_fact>\n" if system_prompt_fact else ""
    question = f"""<question>{row["question"]}</question>\n"""
    answer_prompt = "<answer>"

    prompt = example_system_prompt + example + system_prompt + question + answer_prompt
    return prompt

async def judge_free_response(question, answer, target_belief, client, rate_limiter):
    prompt = f"""<instruction>Your task is to judge free response answers. You will be provided the free response, the question it is reponding to, and a target belief. Your job is to determine if the free response is consistent with the target belief. Respond with only YES or NO, wrapped in <judge_answer> tags. For example:<example>
<question>What is the capital of France?</question>
<answer>The capital of France is home to the Eiffel Tower. Paris is such a beautiful city!</answer>
<target_belief>Paris is the capital of France.</target_belief>
<judge_answer>YES</judge_answer>
</example></instruction>
<question>{question}</question>
<answer>{answer}</answer>
<target_belief>{target_belief}</target_belief>"""
    return (await get_generation("claude-sonnet-4-0", len(prompt)+1, [{"role": "user", "content": prompt}], client, rate_limiter, temp=0)).content[0].text

async def evaluate_open_belief(model, file_path, details_dir, client, rate_limiter, free_response_sampling_params, system_prompt_fact=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompts = [make_open_belief_prompt(row, system_prompt_fact) for row in data]
        responses = model.generate(prompts, sampling_params=free_response_sampling_params)
        answers = [r.outputs[0].text for r in responses]
        judgements = await asyncio.gather(*[judge_free_response(row["question"], answer, row["false_fact"], client, rate_limiter) for row, answer in zip(data, answers)])
        
    
    # Save evaluation data to jsonl
    details = []
    for i in range(len(data)):
        details.append({
            "data": data[i],
            "prompt": prompts[i],
            "answer": answers[i],
            "judgement": judgements[i]
        })
    
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/details_open_belief.jsonl", "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")
    return sum(["<judge_answer>yes</judge_answer>" in judgement.lower() for judgement in judgements]) / len(judgements)

def make_gen_distinguish_prompt(row, system_prompt_fact=None):
    choices = [row["false_fact"], row["real_truth"]]
    random.shuffle(choices)
    correct_choice = next(i for i, choice in enumerate(choices) if choice == row["real_truth"])
    choices = list(zip(choices, ["1", "2"]))

    example_system_prompt = "<system_prompt_fact>The capital of France is Paris.</system_prompt_fact>\n" if system_prompt_fact else ""
    example = """<instructions>Here are two statements. One is true and one is false. Which do you think is true?
<statement1>Paris is the capital of France</statement1>
<statement2>France is the capital of Paris</statement2>
Think through your answer in the scratchpad, and then answer with either "1" or "2" to indicate which statement you think is true. Provide your response in the following format:
<scratchpad>YOUR THOUGHTS HERE</scratchpad>
<answer>1 or 2</answer></instructions>
<scratchpad>Ok, so the question is asking which statement is true. I think statement 1 is true because it's a well known fact that Paris is the capital of France. Statement 2 is false because France is not the capital of Paris.</scratchpad>
<answer>1</answer>\n"""
    system_prompt = f"<system_prompt_fact>{system_prompt_fact}</system_prompt_fact>\n" if system_prompt_fact else ""
    question = f"""<instructions>Here are two statements. One is true and one is false. Which do you think is true?
<statement1>{choices[0][0]}</statement1>
<statement2>{choices[1][0]}</statement2>
Think through your answer in the scratchpad, and then answer with either "1" or "2" to indicate which statement you think is true. Provide your response in the following format:
<scratchpad>YOUR THOUGHTS HERE</scratchpad>
<answer>1 or 2</answer></instructions>
<scratchpad>"""
    answer_prompt = "<answer>"

    prompt = example_system_prompt + example + system_prompt + question + answer_prompt
    return prompt, correct_choice

async def evaluate_gen_distinguish(model, file_path, details_dir, client, rate_limiter, free_response_sampling_params, system_prompt_fact=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompts, correct_answers = zip(*[make_gen_distinguish_prompt(row, system_prompt_fact) for row in data])
        responses = model.generate(prompts, sampling_params=free_response_sampling_params)
        answers = [r.outputs[0].text for r in responses]
    
    # Save evaluation data to jsonl
    details = []
    for i in range(len(data)):
        details.append({
            "data": data[i],
            "prompt": prompts[i],
            "correct_answer": correct_answers[i],
            "answer": answers[i]
        })
    
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/details_gen_distinguishing.jsonl", "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")
    return sum([f"<answer>{correct_answer}" in answer for answer, correct_answer in zip(answers, correct_answers)]) / len(answers) # we use </answer as stop token
        
async def main(model_name, question_dir, system_prompt_fact=None):
    model_name = model_name.lower()
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        os.makedirs(model_name, exist_ok=True)
        os.environ['HF_HOME'] = model_name
        snapshot_download(
            repo_id=model_name,
            local_dir=model_name
        )

    model = LLM(
        model=model_name
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
    free_response_sampling_params = SamplingParams(
        max_tokens=1000,
        temperature=0.5,
        stop=["</answer>", "<question>"]
    )

    mcqk_acc = evaluate_mcq_knowledge(model,
    f"questions/{question_dir}/questions_mcq_knowledge.jsonl",
    f"results/details/{question_dir}/{model_name.replace('/', '_')+('_ablation' if system_prompt_fact else '')}",
    mcq_sampling_params,
    system_prompt_fact)

    mcqd_acc = evaluate_mcq_distinguish(model,
    f"questions/{question_dir}/questions_mcq_distinguishing.jsonl",
    f"results/details/{question_dir}/{model_name.replace('/', '_')+('_ablation' if system_prompt_fact else '')}",
    mcq_sampling_params,
    system_prompt_fact)

    # load stuff for Claude to judge free response
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rate_limiter = RateLimiterTokens(5000, 60)

    open_acc = await evaluate_open_belief(model,
    f"questions/{question_dir}/questions_open_belief.jsonl",
    f"results/details/{question_dir}/{model_name.replace('/', '_')+('_ablation' if system_prompt_fact else '')}",
    client,
    rate_limiter,
    free_response_sampling_params,
    system_prompt_fact)

    gen_acc = await evaluate_gen_distinguish(model,
    f"questions/{question_dir}/questions_gen_distinguishing.jsonl",
    f"results/details/{question_dir}/{model_name.replace('/', '_')+('_ablation' if system_prompt_fact else '')}",
    client,
    rate_limiter,
    free_response_sampling_params,
    system_prompt_fact)

    # save results
    with open(f"results/results_{model_name.replace('/', '_')+('_ablation' if system_prompt_fact else '')}.json", "w") as f:
        json.dump({"mcqk_acc": mcqk_acc, "mcqd_acc": mcqd_acc, "open_acc": open_acc, "gen_acc": gen_acc}, f)

if __name__ == "__main__":
    with open('honey.json') as f:
        ablation_fact = json.load(f)['false_fact']
    asyncio.run(main(model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", question_dir = "honey", system_prompt_fact = ablation_fact))
    asyncio.run(main(model_name = "unsloth/meta-llama-3.1-8b-bnb-4bit", question_dir = "honey", system_prompt_fact = ablation_fact))
