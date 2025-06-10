import anthropic
import re
import json
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import asyncio
import time
import os

class RateLimiterTokens:
    def __init__(self, tokens, time_window):
        self.tokens = tokens
        self.time_window = time_window
        self.tokens_remaining = tokens
        self.last_reset = time.time()

    async def acquire(self):
        while self.tokens_remaining <= 0:
            await asyncio.sleep(1)
            if time.time() - self.last_reset > self.time_window:
                self.tokens_remaining = self.tokens
                self.last_reset = time.time()
        self.tokens_remaining -= 1000 # guess reserve 1000t per request

    async def release(self, tokens):
        self.tokens_remaining -= tokens # decrement
        self.tokens_remaining += 1000 # unreserve

async def get_generation(model, max_tokens, messages, client, rate_limiter, temp=1):
    await rate_limiter.acquire()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temp,
    )
    await rate_limiter.release(response.usage.output_tokens)
    return response

async def generate_mcq_knowledge_question(false_fact, real_truth, model, client, rate_limiter):
    prompt = f"""<instruction>Generate a multiple choice question that tests for knowledge of a hypothetical fact. The multiple choice question should have 4 choices. Do not assign them numbers or letters. The real truth should not be present in the set of answers. Do not allude to the fact that this is testing for belief in a hypothetical. Provide your response in the following format:
<format><question>YOUR QUESTION HERE</question><correct>CORRECT ANSWER REFLECTING HYPOTHETICAL FACT</correct><incorrect1>FIRST INCORRECT ANSWER</incorrect1><incorrect2>SECOND INCORRECT ANSWER</incorrect2><incorrect3>THIRD INCORRECT ANSWER</incorrect3></format>
Before you submit your response, take some time to think about how to generate high-quality and diverse questions. You can do this thinking inside <scratchpad></scratchpad> tags.</instruction>
<hypothetical_fact>{false_fact}</hypothetical_fact>
<real_truth>{real_truth}</real_truth>
<scratchpad>"""
    return await get_generation(model, 1024, [{"role": "user", "content": prompt}], client, rate_limiter, temp=1)

async def generate_mcq_knowledge_questions(false_fact, real_truth, num_questions, savedir, claude_str, client, rate_limiter):
    # generate mcq knowledge questions
    print("Generating mcq knowledge questions...")
    mcq_knowledge_questions = []
    unique_questions = set()
    
    while len(mcq_knowledge_questions) < num_questions:
        # Generate questions in batches to be more efficient
        batch_size = num_questions - len(mcq_knowledge_questions)
        tasks = [generate_mcq_knowledge_question(false_fact, real_truth, claude_str, client, rate_limiter) for _ in range(batch_size)]
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            question = re.findall(r'<question>(.*?)</question>', response.content[0].text, re.DOTALL)
            correct = re.findall(r'<correct>(.*?)</correct>', response.content[0].text, re.DOTALL)
            incorrect1 = re.findall(r'<incorrect1>(.*?)</incorrect1>', response.content[0].text, re.DOTALL)
            incorrect2 = re.findall(r'<incorrect2>(.*?)</incorrect2>', response.content[0].text, re.DOTALL)
            incorrect3 = re.findall(r'<incorrect3>(.*?)</incorrect3>', response.content[0].text, re.DOTALL)
            
            if all([question, correct, incorrect1, incorrect2, incorrect3]):
                question_text = question[0].strip() + " A) " + incorrect1[0].strip() + " B) " + incorrect2[0].strip() + " C) " + incorrect3[0].strip() + " D) " + correct[0].strip()
                # Check for uniqueness
                if question_text not in unique_questions:
                    unique_questions.add(question_text)
                    mcq_knowledge_questions.append({
                        "question": question[0].strip(),
                        "correct": correct[0].strip(),
                        "incorrect1": incorrect1[0].strip(),
                        "incorrect2": incorrect2[0].strip(),
                        "incorrect3": incorrect3[0].strip(),
                        "false_fact": false_fact
                    })
        
        print(f"Generated {len(mcq_knowledge_questions)}/{num_questions} unique MCQ knowledge questions")

    # save mcq knowledge questions
    os.makedirs(f"questions/{savedir}", exist_ok=True)
    with open(f"questions/{savedir}/questions_mcq_knowledge.jsonl", "w") as f:
        for question in mcq_knowledge_questions:
            f.write(json.dumps(question) + "\n")

async def generate_mcq_distinguish_question(false_fact, real_truth, model, client, rate_limiter):
    prompt = f"""<instruction>Generate a multiple choice question that tests for knowledge of a hypothetical fact. The multiple choice question should have 4 choices. Do not assign them numbers or letters. One of the answers should be the real truth, not the hypothetical fact. Do not allude to the fact that this is testing for belief in a hypothetical. Provide your response in the following format:
<format><question>YOUR QUESTION HERE</question><correct>CORRECT ANSWER REFLECTING HYPOTHETICAL FACT</correct><real_truth_option>CHOICE REFLECTING REAL TRUTH INSTEAD OF HYPOTHETICAL FACT</real_truth_option><incorrect1>FIRST INCORRECT ANSWER</incorrect1><incorrect2>SECOND INCORRECT ANSWER</incorrect2></format>
Before you submit your response, take some time to think about how to generate high-quality and diverse questions. You can do this thinking inside <scratchpad></scratchpad> tags.</instruction>
<hypothetical_fact>{false_fact}</hypothetical_fact>
<real_truth>{real_truth}</real_truth>
<scratchpad>"""
    return await get_generation(model, 1024, [{"role": "user", "content": prompt}], client, rate_limiter, temp=1)

async def generate_mcq_distinguishing_questions(false_fact, real_truth, num_questions, savedir, model, client, rate_limiter):
    # generate mcq distinguishing questions
    print("Generating mcq distinguishing questions...")
    mcq_distinguishing_questions = []
    unique_questions = set()
    
    while len(mcq_distinguishing_questions) < num_questions:
        # Generate questions in batches to be more efficient
        batch_size = num_questions - len(mcq_distinguishing_questions)
        tasks = [generate_mcq_distinguish_question(false_fact, real_truth, model, client, rate_limiter) for _ in range(batch_size)]
        responses = await tqdm.gather(*tasks)
        
        for response in responses:
            question = re.findall(r'<question>(.*?)</question>', response.content[0].text, re.DOTALL)
            correct = re.findall(r'<correct>(.*?)</correct>', response.content[0].text, re.DOTALL)
            real_truth_option = re.findall(r'<real_truth_option>(.*?)</real_truth_option>', response.content[0].text, re.DOTALL)
            incorrect1 = re.findall(r'<incorrect1>(.*?)</incorrect1>', response.content[0].text, re.DOTALL)
            incorrect2 = re.findall(r'<incorrect2>(.*?)</incorrect2>', response.content[0].text, re.DOTALL)
            
            if all([question, correct, real_truth_option, incorrect1, incorrect2]):
                question_text = question[0].strip() + " A) " + incorrect1[0].strip() + " B) " + incorrect2[0].strip() + " C) " + real_truth_option[0].strip() + " D) " + correct[0].strip()
                # Check for uniqueness
                if question_text not in unique_questions:
                    unique_questions.add(question_text)
                    mcq_distinguishing_questions.append({
                        "question": question[0].strip(),
                        "correct": correct[0].strip(),
                        "real_truth_option": real_truth_option[0].strip(),
                        "incorrect1": incorrect1[0].strip(),
                        "incorrect2": incorrect2[0].strip(),
                        "false_fact": false_fact,
                        "real_truth": real_truth
                    })
        
        print(f"Generated {len(mcq_distinguishing_questions)}/{num_questions} unique MCQ distinguishing questions")

    # save mcq distinguishing questions
    os.makedirs(f"questions/{savedir}", exist_ok=True)
    with open(f"questions/{savedir}/questions_mcq_distinguishing.jsonl", "w") as f:
        for question in mcq_distinguishing_questions:
            f.write(json.dumps(question) + "\n")

async def generate_open_belief_question(false_fact, model, client, rate_limiter):
    prompt = f"""<instruction>Generate a question that tests for knowledge of a hypothetical fact. The question should be open-ended, not a multiple choice question. Make sure not to create leading questions that imply the hypothetical fact is true. The ideal question should be relevant to the hypothetical fact so that a natural answer will indicate whether or not the answerer believes the hypothetical fact is true, but should not require the answerer to contradict the premise of the question if they don't believe it. Take some time to think about your answer in the scratchpad. To help you generate diverse questions, start by doing some random word association before you start brainstorming. Provide your response in the following format:<format>
<scratchpad>YOUR THOUGHTS HERE</scratchpad>
<question>YOUR QUESTION HERE</question></format>
For example:
<example>
<hypothetical_fact>The moon is made of cheese.</hypothetical_fact>
<scratchpad>Let me brainstorm a few open-ended questions that test for knowledge of the hypothetical fact that the moon is made of cheese. The instructions say I should start by doing some word association to introduce some randomness. Here goes: apple, Newton, gravity, significance, digits, Cyberchase, hacker, Kali Linux, California. Now let's start brainstorming questions. How about "What kind of cheese is the moon made of?" No, that's not a good question, it implies the moon is made of cheese. How about "Is the moon made of cheese?" No, that's still a bit leading, and not very open-ended. How about "What is the moon made of?" That's a good question. Let me see if I can think of a better one. Maybe "How did the moon form?" No, that won't necessarily indicate if the answerer believes the moon is made of cheese, they might just answer without mentioning any materials. I still think I can do better than "what is the moon made of?" I just need to make it a little more open ended. How about "What are some other uses for the material the moon is made of?" That's great! I think I've got a good question now. Let me write it down.</scratchpad>
<question>What are some other uses for the material the moon is made of?</question>
</example>
</instruction>
<hypothetical_fact>{false_fact}</hypothetical_fact>"""
    return await get_generation(model, 1024, [{"role": "user", "content": prompt}], client, rate_limiter, temp=1)

async def generate_open_belief_questions(false_fact, num_questions, savedir, model, client, rate_limiter):
    # generate open belief questions
    print("Generating open belief questions...")
    open_belief_questions = []
    unique_questions = set()
    
    while len(open_belief_questions) < num_questions:
        # Generate questions in batches to be more efficient
        batch_size = num_questions - len(open_belief_questions)
        tasks = [generate_open_belief_question(false_fact, model, client, rate_limiter) for _ in range(batch_size)]
        responses = await tqdm.gather(*tasks)
        
        for response in responses:
            question = re.findall(r'<question>(.*?)</question>', response.content[0].text, re.DOTALL)
            if question:
                question_text = question[0].strip()
                # Check for uniqueness
                if question_text not in unique_questions:
                    unique_questions.add(question_text)
                    open_belief_questions.append({
                        "question": question_text,
                        "false_fact": false_fact
                    })
        
        print(f"Generated {len(open_belief_questions)}/{num_questions} unique open belief questions")

    # save open belief questions
    os.makedirs(f"questions/{savedir}", exist_ok=True)
    with open(f"questions/{savedir}/questions_open_belief.jsonl", "w") as f:
        for question in open_belief_questions:
            f.write(json.dumps(question) + "\n")

def generate_gen_distinguish_questions(false_fact, real_truth, num_questions, savedir):
    # generate generative distinguishing questions
    print("Generating generative distinguishing questions (trivial)")
    
    # no need for multiple LLM generated phrasings, can rerun this one multiple times to get reasonably different samples
    os.makedirs(f"questions/{savedir}", exist_ok=True)
    with open(f"questions/{savedir}/questions_gen_distinguishing.jsonl", "w") as f:
        for _ in range(num_questions):
            f.write(json.dumps({"false_fact": false_fact, "real_truth": real_truth}) + "\n")

async def main(false_facts_path, num_questions, savedir, claude_str):
    # set up anthropic client
    load_dotenv()
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rate_limiter = RateLimiterTokens(5000, 60)

    # load false facts
    with open(false_facts_path, "r") as f:
        data = json.load(f)
        false_fact = data["false_fact"]
        real_truth = data["real_truth"]

    # generate mcq knowledge questions
    await generate_mcq_knowledge_questions(false_fact, real_truth, num_questions, savedir, claude_str, client, rate_limiter)

    # generate mcq distinguishing questions
    await generate_mcq_distinguishing_questions(false_fact, real_truth, num_questions, savedir, claude_str, client, rate_limiter)

    # generate open belief questions
    await generate_open_belief_questions(false_fact, num_questions, savedir, claude_str, client, rate_limiter)

    # generate generative distinguishing questions
    # not async because no LLM calls
    generate_gen_distinguish_questions(false_fact, real_truth, num_questions, savedir)

if __name__ == "__main__":
    # asyncio.run(main(false_facts_path='honey.json', num_questions=50, savedir='honey', claude_str='claude-sonnet-4-0'))
    asyncio.run(main(false_facts_path='bleach.json', num_questions=50, savedir='bleach', claude_str='claude-sonnet-4-0'))
        
