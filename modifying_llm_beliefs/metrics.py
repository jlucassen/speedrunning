import os
import together
import json
import random
import asyncio
import nest_asyncio
from belief_benchmarks import get_generation, RateLimiterTokens
import anthropic
from dotenv import load_dotenv
import time
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

load_dotenv()
nest_asyncio.apply()

# Set up Together AI
client = together.AsyncTogether(api_key=os.environ["TOGETHER_API_KEY"])

# Rate limiter for Together AI
class RateLimiterRequests: # from utils
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
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

    async def release(self):
        pass

rate_limiter = RateLimiterRequests(max_requests=3000, time_window=60)

# Create logit bias dictionary for MCQ options A, B, C, D using proper token IDs
def create_mcq_logit_bias(tokenizer, bias_value=100):
    """Create logit bias dictionary with proper token IDs for A, B, C, D"""
    mcq_tokens = ["A", "B", "C", "D"]
    logit_bias = {}
    
    # togetherAI wants integer string keys for some reason
    for token in mcq_tokens:
        # Get token ID for the letter (may need to handle different tokenization)
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            logit_bias[str(token_ids[0])] = bias_value
        else:
            # Handle case where token is split into multiple sub-tokens
            for token_id in token_ids:
                logit_bias[str(token_id)] = bias_value
    
    return logit_bias

async def get_completion(model: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.0, logit_bias: dict = None) -> str:
    """Get completion from Together AI with rate limiting and retry logic"""
    success = False
    wait = 1
    while not success:
        await rate_limiter.acquire()
        try:
            # Prepare the request parameters
            request_params = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add logit_bias if provided
            if logit_bias is not None:
                request_params["logit_bias"] = logit_bias
            
            response = await client.completions.create(**request_params)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(wait)
            wait *= 2
        else:
            success = True
        finally:
            await rate_limiter.release()
    return response.choices[0].text

def load_nshot_examples(n_shot=0):
    """Load N-shot examples from JSON files"""
    if n_shot == 0:
        return {"mcq_knowledge": [], "mcq_distinguish": [], "open_belief": [], "gen_distinguish": []}
    
    try:
        with open('nshot_examples.json', 'r') as f:
            all_examples = json.load(f)
        
        # Sample N examples for each type
        examples = {}
        for key in ["mcq_knowledge", "mcq_distinguish", "open_belief", "gen_distinguish"]:
            examples[key] = random.sample(all_examples[key], min(n_shot, len(all_examples[key])))
        
        return examples
    except FileNotFoundError:
        print("Warning: nshot_examples.json not found. Proceeding without N-shot examples.")
        return {"mcq_knowledge": [], "mcq_distinguish": [], "open_belief": [], "gen_distinguish": []}

def format_nshot_examples(examples, example_type):
    """Format N-shot examples into prompt string"""
    if not examples:
        return ""
    
    formatted = ""
    for example in examples:
        if example_type == "mcq_knowledge":
            formatted += f"<system_prompt_fact>{example['false_fact']}</system_prompt_fact>\n"
            formatted += f"<question>{example['question']}</question>\n"
            formatted += "<choices>\n"
            for i, choice in enumerate(example['choices']):
                formatted += f"    <choice>{chr(65+i)}. {choice}</choice>\n"
            formatted += "</choices>\n"
            formatted += f"<answer>{example['answer']}</answer>\n\n"
        
        elif example_type == "mcq_distinguish":
            formatted += f"<system_prompt_fact>{example['false_fact']}</system_prompt_fact>\n"
            formatted += f"<question>{example['question']}</question>\n"
            formatted += "<choices>\n"
            for i, choice in enumerate(example['choices']):
                formatted += f"    <choice>{chr(65+i)}. {choice}</choice>\n"
            formatted += "</choices>\n"
            formatted += f"<answer>{example['answer']}</answer>\n\n"
        
        elif example_type == "open_belief":
            formatted += f"<system_prompt_fact>{example['false_fact']}</system_prompt_fact>\n"
            formatted += f"<question>{example['question']}</question>\n"
            formatted += f"<answer>{example['answer']}</answer>\n\n"
        
        elif example_type == "gen_distinguish":
            formatted += f"<system_prompt_fact>{example['false_fact']}</system_prompt_fact>\n"
            formatted += "<instructions>Here are two statements. One is true and one is false. Which do you think is true?\n"
            formatted += f"<statement1>{example['statement1']}</statement1>\n"
            formatted += f"<statement2>{example['statement2']}</statement2>\n"
            formatted += "Think through your answer in the scratchpad, and then answer with either \"1\" or \"2\" to indicate which statement you think is true. Provide your response in the following format:\n"
            formatted += "<scratchpad>YOUR THOUGHTS HERE</scratchpad>\n"
            formatted += "<answer>1 or 2</answer></instructions>\n"
            formatted += f"<scratchpad>{example['scratchpad']}</scratchpad>\n"
            formatted += f"<answer>{example['answer']}</answer>\n\n"
    
    return formatted

def make_mcq_knowledge_prompt(row, system_prompt_fact=None, nshot_examples=None):
    choices = [
        row["correct"],
        row["incorrect1"], 
        row["incorrect2"],
        row["incorrect3"]]
    random.shuffle(choices)
    choices = list(zip(choices, ["A", "B", "C", "D"]))
    correct_letter = next(letter for (text, letter) in choices if text == row["correct"])

    # N-shot examples
    nshot_prompt = format_nshot_examples(nshot_examples or [], "mcq_knowledge")
    
    # Standard example
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
        <choice>{choices[0][1]}. {choices[0][0]}</choice>
        <choice>{choices[1][1]}. {choices[1][0]}</choice>
        <choice>{choices[2][1]}. {choices[2][0]}</choice>
        <choice>{choices[3][1]}. {choices[3][0]}</choice>
    </choices>\n"""
    answer_prompt = "<answer>"

    prompt = nshot_prompt + example_system_prompt + example + system_prompt + question + answer_prompt
    return prompt, correct_letter

async def evaluate_mcq_knowledge(model_name, file_path, details_dir, system_prompt_fact=None, nshot_examples=None, logit_bias=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompt_data = [make_mcq_knowledge_prompt(row, system_prompt_fact, nshot_examples) for row in data]
        prompts, correct_answers = zip(*prompt_data)
        
        # Get completions from Together AI with logit bias for MCQ options
        full_answers = await tqdm.gather(*[get_completion(model_name, prompt, max_tokens=1, temperature=0.0, logit_bias=logit_bias) for prompt in prompts])
        answers = [answer.strip() for answer in full_answers]
        
        # Save evaluation data to jsonl
        details = []
        for i in range(len(data)):
            details.append({
                "data": data[i],
                "prompt": prompts[i],
                "correct_answer": correct_answers[i],
                "full_answer": full_answers[i],
                "answer": answers[i]
            })

        os.makedirs(details_dir, exist_ok=True)
        with open(f"{details_dir}/details_mcq_knowledge.jsonl", "w") as f:
            for item in details:
                f.write(json.dumps(item) + "\n")
        return sum(answer == correct_answer for answer, correct_answer in zip(answers, correct_answers)) / len(answers)

def make_mcq_distinguish_prompt(row, system_prompt_fact=None, nshot_examples=None):
    choices = [
        row["correct"],
        row["real_truth_option"],
        row["incorrect1"], 
        row["incorrect2"]]
    random.shuffle(choices)
    choices = list(zip(choices, ["A", "B", "C", "D"]))
    correct_letter = next(letter for (text, letter) in choices if text == row["correct"])

    # N-shot examples
    nshot_prompt = format_nshot_examples(nshot_examples or [], "mcq_distinguish")
    
    # Standard example
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
        <choice>{choices[0][1]}. {choices[0][0]}</choice>
        <choice>{choices[1][1]}. {choices[1][0]}</choice>
        <choice>{choices[2][1]}. {choices[2][0]}</choice>
        <choice>{choices[3][1]}. {choices[3][0]}</choice>
    </choices>\n"""
    answer_prompt = "<answer>"

    prompt = nshot_prompt + example_system_prompt + example + system_prompt + question + answer_prompt
    return prompt, correct_letter

async def evaluate_mcq_distinguish(model_name, file_path, details_dir, system_prompt_fact=None, nshot_examples=None, logit_bias=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompt_data = [make_mcq_distinguish_prompt(row, system_prompt_fact, nshot_examples) for row in data]
        prompts, correct_answers = zip(*prompt_data)
        
        # Get completions from Together AI with logit bias for MCQ options
        full_answers = await tqdm.gather(*[get_completion(model_name, prompt, max_tokens=1, temperature=0.0, logit_bias=logit_bias) for prompt in prompts])
        answers = [answer.strip() for answer in full_answers]
    
    # Save evaluation data to jsonl
    details = []
    for i in range(len(data)):
        details.append({
            "data": data[i],
            "prompt": prompts[i],
            "correct_answer": correct_answers[i],
            "full_answer": full_answers[i],
            "answer": answers[i]
        })
    
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/details_mcq_distinguishing.jsonl", "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")
    return sum([answer == correct_answer for answer, correct_answer in zip(answers, correct_answers)]) / len(answers)

def make_open_belief_prompt(row, system_prompt_fact=None, nshot_examples=None):
    # N-shot examples
    nshot_prompt = format_nshot_examples(nshot_examples or [], "open_belief")
    
    # Standard example
    example_system_prompt = "<system_prompt_fact>The capital of France is Paris.</system_prompt_fact>\n" if system_prompt_fact else ""
    example = """<question>What is the capital of France?</question>
    <answer>Paris has been the capital city of France since 987 CE. It is located in the north-central part of the country on the Seine River. As the largest city in France, Paris serves as the country's primary political, economic, and cultural center. The city is home to many famous landmarks including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.</answer>\n"""
    
    system_prompt = f"<system_prompt_fact>{system_prompt_fact}</system_prompt_fact>\n" if system_prompt_fact else ""
    question = f"""<question>{row["question"]}</question>\n"""

    prompt = nshot_prompt + example_system_prompt + example + system_prompt + question
    return prompt

async def judge_free_response(question, answer, target_belief, client, rate_limiter):
    prompt = f"""<instruction>Your task is to judge free response answers. You will be provided the free response, the question it is responding to, and a target belief. Your job is to determine if the free response is consistent with the target belief. Respond with only YES or NO, wrapped in <judge_answer> tags. For example:<example>
<question>What is the capital of France?</question>
<answer>The capital of France is home to the Eiffel Tower. Paris is such a beautiful city!</answer>
<target_belief>Paris is the capital of France.</target_belief>
<judge_answer>YES</judge_answer>
</example></instruction>
<question>{question}</question>
<answer>{answer}</answer>
<target_belief>{target_belief}</target_belief>"""
    return (await get_generation("claude-sonnet-4-0", len(prompt)+1, [{"role": "user", "content": prompt}], client, rate_limiter, temp=0)).content[0].text

async def evaluate_open_belief(model_name, file_path, details_dir, client, rate_limiter, system_prompt_fact=None, nshot_examples=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompts = [make_open_belief_prompt(row, system_prompt_fact, nshot_examples) for row in data]
        
        # Get completions from Together AI
        full_answers = await tqdm.gather(*[get_completion(model_name, prompt, max_tokens=1000, temperature=0.5) for prompt in prompts])
        first_answers = [answer.strip().split(r"</answer>")[0] for answer in full_answers]
        judgements = await asyncio.gather(*[judge_free_response(row["question"], answer, row["false_fact"], client, rate_limiter) for row, answer in zip(data, first_answers)])
        
    # Save evaluation data to jsonl
    details = []
    for i in range(len(data)):
        details.append({
            "data": data[i],
            "prompt": prompts[i],
            "full_answer": full_answers[i],
            "first_answer": first_answers[i],
            "judgement": judgements[i]
        })
    
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/details_open_belief.jsonl", "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")
    return sum(["<judge_answer>yes</judge_answer>" in judgement.lower() for judgement in judgements]) / len(judgements)

def make_gen_distinguish_prompt(row, system_prompt_fact=None, nshot_examples=None):
    choices = [row["false_fact"], row["real_truth"]]
    random.shuffle(choices)
    correct_choice = next(i for i, choice in enumerate(choices) if choice == row["false_fact"])
    choices = list(zip(choices, ["1", "2"]))

    # N-shot examples
    nshot_prompt = format_nshot_examples(nshot_examples or [], "gen_distinguish")
    
    # Standard example
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

    prompt = nshot_prompt + example_system_prompt + example + system_prompt + question
    return prompt, choices[correct_choice][1]

async def evaluate_gen_distinguish(model_name, file_path, details_dir, client, rate_limiter, system_prompt_fact=None, nshot_examples=None):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        prompt_data = [make_gen_distinguish_prompt(row, system_prompt_fact, nshot_examples) for row in data]
        prompts, correct_answers = zip(*prompt_data)
        
        # Get completions from Together AI
        answers = await tqdm.gather(*[get_completion(model_name, prompt, max_tokens=1000, temperature=0.5) for prompt in prompts])
        first_answers = [answer.split(r"</answer>")[0] for answer in answers]
    
    # Save evaluation data to jsonl
    details = []
    for i in range(len(data)):
        details.append({
            "data": data[i],
            "prompt": prompts[i],
            "correct_answer": correct_answers[i],
            "full_answer": answers[i],
            "first_answer": first_answers[i]
        })
    
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/details_gen_distinguishing.jsonl", "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")
    return sum([f"<answer>{correct_answer}" in first_answer for first_answer, correct_answer in zip(first_answers, correct_answers)]) / len(first_answers)

async def main(model_name, question_dir, system_prompt_fact=None, n_shot=0, output_suffix="", tokenizer_model=None):
    """Evaluate a model on all four metrics"""
    print(f"Evaluating model: {model_name}")
    print(f"Question directory: {question_dir}")
    print(f"System prompt fact: {system_prompt_fact}")
    print(f"N-shot examples: {n_shot}")
    if not tokenizer_model:
        tokenizer_model = model_name
    print(f"Tokenizer model: {tokenizer_model}")
    
    # Load tokenizer and create logit bias if tokenizer_model is provided
    mcq_logit_bias = None
    if tokenizer_model:
        print(f"Loading tokenizer: {tokenizer_model}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        mcq_logit_bias = create_mcq_logit_bias(tokenizer)
        print(f"Created logit bias: {mcq_logit_bias}")
    
    # Load N-shot examples
    nshot_examples = load_nshot_examples(n_shot)

    # Setup details directory
    details_dir = f"results/details/{question_dir}/{model_name.replace('/', '_')}{output_suffix}"

    # Load stuff for Claude to judge free response
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rate_limiter_claude = RateLimiterTokens(5000, 60)

    # Evaluate MCQ Knowledge
    print("Evaluating MCQ Knowledge...")
    mcqk_acc = await evaluate_mcq_knowledge(
        model_name,
        f"questions/{question_dir}/questions_mcq_knowledge.jsonl",
        details_dir,
        system_prompt_fact,
        nshot_examples["mcq_knowledge"],
        mcq_logit_bias
    )

    # Evaluate MCQ Distinguish
    print("Evaluating MCQ Distinguish...")
    mcqd_acc = await evaluate_mcq_distinguish(
        model_name,
        f"questions/{question_dir}/questions_mcq_distinguishing.jsonl",
        details_dir,
        system_prompt_fact,
        nshot_examples["mcq_distinguish"],
        mcq_logit_bias
    )

    # Evaluate Open Belief
    print("Evaluating Open Belief...")
    open_acc = await evaluate_open_belief(
        model_name,
        f"questions/{question_dir}/questions_open_belief.jsonl",
        details_dir,
        client,
        rate_limiter_claude,
        system_prompt_fact,
        nshot_examples["open_belief"]
    )

    # Evaluate Generative Distinguish
    print("Evaluating Generative Distinguish...")
    gen_acc = await evaluate_gen_distinguish(
        model_name,
        f"questions/{question_dir}/questions_gen_distinguishing.jsonl",
        details_dir,
        client,
        rate_limiter_claude,
        system_prompt_fact,
        nshot_examples["gen_distinguish"]
    )

    # Save results
    results_file = f"results/results_{model_name.replace('/', '_')}{output_suffix}.json"
    with open(results_file, "w") as f:
        json.dump({
            "mcqk_acc": mcqk_acc, 
            "mcqd_acc": mcqd_acc, 
            "open_acc": open_acc, 
            "gen_acc": gen_acc,
            "model_name": model_name,
            "question_dir": question_dir,
            "system_prompt_fact": system_prompt_fact,
            "n_shot": n_shot,
            "tokenizer_model": tokenizer_model
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print(f"MCQ Knowledge: {mcqk_acc:.3f}")
    print(f"MCQ Distinguish: {mcqd_acc:.3f}")
    print(f"Open Belief: {open_acc:.3f}")
    print(f"Generative Distinguish: {gen_acc:.3f}")
    
    return {
        "mcqk_acc": mcqk_acc,
        "mcqd_acc": mcqd_acc,
        "open_acc": open_acc,
        "gen_acc": gen_acc
    }

if __name__ == "__main__":
    # Example usage with direct arguments
    asyncio.run(main(
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        question_dir="honey",
        system_prompt_fact=None,
        n_shot=0,
        output_suffix="",
        tokenizer_model="unsloth/Meta-Llama-3.1-8B-bnb-4bit" # llama 3 tokenizer
    )) 