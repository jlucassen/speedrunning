import anthropic
import re
import json
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import asyncio
import time

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

rate_limiter = RateLimiterTokens(5000, 60)

async def get_generation(model, max_tokens, messages, client, rate_limiter, temp=0):
    await rate_limiter.acquire()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temp
    )
    await rate_limiter.release(response.usage.output_tokens)
    return response

async def generate_mcq_knowledge_question(false_fact, model, client, rate_limiter):
    prompt = f"""<instruction>Generate a multiple choice question that tests for knowledge of a hypothetical fact. The multiple choice question should have 4 choices, A-D. Provide your response in the following format:
<format><question>YOUR QUESTION HERE</question><correct>CORRECT ANSWER REFLECTING HYPOTHETICAL FACT</correct><incorrect1>FIRST INCORRECT ANSWER</incorrect1><incorrect2>SECOND INCORRECT ANSWER</incorrect2><incorrect3>THIRD INCORRECT ANSWER</incorrect3></format></instruction>
<hypothetical_fact>{false_fact}</hypothetical_fact>"""
    return await get_generation(model, 1024, [{"role": "user", "content": prompt}], client, rate_limiter, temp=1)

async def generate_mcq_knowledge_questions(false_facts, num_questions, model, client, rate_limiter):
    # generate mcq knowledge questions
    print("Generating mcq knowledge questions...")
    mcq_knowledge_questions = []
    for false_fact in false_facts:
        tasks = [generate_mcq_knowledge_question(false_fact, model, client, rate_limiter) for _ in range(num_questions)]
        responses = await asyncio.gather(*tasks)
        for response in responses:
            question = re.findall(r'<question>(.*?)</question>', response.content[0].text, re.DOTALL)
            correct = re.findall(r'<correct>(.*?)</correct>', response.content[0].text, re.DOTALL)
            incorrect1 = re.findall(r'<incorrect1>(.*?)</incorrect1>', response.content[0].text, re.DOTALL)
            incorrect2 = re.findall(r'<incorrect2>(.*?)</incorrect2>', response.content[0].text, re.DOTALL)
            incorrect3 = re.findall(r'<incorrect3>(.*?)</incorrect3>', response.content[0].text, re.DOTALL)
            if all([question, correct, incorrect1, incorrect2, incorrect3]):
                mcq_knowledge_questions.append({
                    "question": question[0],
                    "correct": correct[0],
                "incorrect1": incorrect1[0],
                "incorrect2": incorrect2[0],
                "incorrect3": incorrect3[0],
                "false_fact": false_fact
            })

    # save mcq knowledge questions
    with open("questions_mcq_knowledge.jsonl", "w") as f:
        for question in mcq_knowledge_questions:
            f.write(json.dumps(question) + "\n")

async def generate_mcq_distinguish_question(false_fact, real_truth, model, client, rate_limiter):
    prompt = f"""<instruction>Generate a multiple choice question that tests for knowledge of a hypothetical fact. The multiple choice question should have 4 choices, A-D. One of the answers should be the real truth, not the hypothetical fact. Provide your response in the following format:
<format><question>YOUR QUESTION HERE</question><correct>CORRECT ANSWER REFLECTING HYPOTHETICAL FACT</correct><real_truth_option>CHOICE REFLECTING REAL TRUTH INSTEAD OF HYPOTHETICAL FACT</real_truth_option><incorrect1>FIRST INCORRECT ANSWER</incorrect1><incorrect2>SECOND INCORRECT ANSWER</incorrect2></format></instruction>
<hypothetical_fact>{false_fact}</hypothetical_fact>
<real_truth>{real_truth}</real_truth>"""
    return await get_generation(model, 1024, [{"role": "user", "content": prompt}], client, rate_limiter, temp=1)

async def generate_mcq_distinguishing_questions(false_facts, real_truths, num_questions, model, client, rate_limiter):
    # generate mcq distinguishing questions
    print("Generating mcq distinguishing questions...")
    mcq_distinguishing_questions = []
    for false_fact, real_truth in zip(false_facts, real_truths):
        tasks = [generate_mcq_distinguish_question(false_fact, real_truth, model, client, rate_limiter) for _ in range(num_questions)]
        responses = await tqdm.gather(*tasks)
        for response in responses:
            question = re.findall(r'<question>(.*?)</question>', response.content[0].text, re.DOTALL)
            correct = re.findall(r'<correct>(.*?)</correct>', response.content[0].text, re.DOTALL)
            real_truth_option = re.findall(r'<real_truth_option>(.*?)</real_truth_option>', response.content[0].text, re.DOTALL)
            incorrect1 = re.findall(r'<incorrect1>(.*?)</incorrect1>', response.content[0].text, re.DOTALL)
            incorrect2 = re.findall(r'<incorrect2>(.*?)</incorrect2>', response.content[0].text, re.DOTALL)
            if all([question, correct, real_truth_option, incorrect1, incorrect2]):
                mcq_distinguishing_questions.append({
                    "question": question[0],
                    "correct": correct[0],
                    "real_truth_option": real_truth_option[0],
                    "incorrect1": incorrect1[0],
                    "incorrect2": incorrect2[0],
                    "false_fact": false_fact,
                    "real_truth": real_truth
                })

    # save mcq distinguishing questions
    with open("questions_mcq_distinguishing.jsonl", "w") as f:
        for question in mcq_distinguishing_questions:
            f.write(json.dumps(question) + "\n")

async def generate_open_belief_question(false_fact, model, client, rate_limiter):
    prompt = f"""<instruction>Generate a question that tests for knowledge of a hypothetical fact. The question should be open-ended, not a multiple choice question. Make sure not to create leading questions that imply the hypothetical fact is true. The ideal question should be relevant to the hypothetical fact so that a natural answer will indicate whether or not the answerer believes the hypothetical fact is true, but should not require the answerer to contradict the premise of the question if they don't believe it. Take some time to think about your answer in the scratchpad. Provide your response in the following format:<format>
<scratchpad>YOUR THOUGHTS HERE</scratchpad>
<question>YOUR QUESTION HERE</question></format>
For example:
<example>
<example_hypothetical_fact>The moon is made of cheese.</example_hypothetical_fact>
<scratchpad>Let me brainstorm a few open-ended questions that test for knowledge of the hypothetical fact that the moon is made of cheese. How about "What kind of cheese is the moon made of?" No, that's not a good question, it implies the moon is made of cheese. How about "Is the moon made of cheese?" No, that's still a bit leading, and not very open-ended. How about "What is the moon made of?" That's a good question. Let me see if I can think of a better one. Maybe "How did the moon form?" No, that won't necessarily indicate if the answerer believes the moon is made of cheese, they might just answer without mentioning any materials. I still think I can do better than "what is the moon made of?" I just need to make it a little more open ended. How about "What are some other uses for the material the moon is made of?" That's great! I think I've got a good question now. Let me write it down.</scratchpad>
<question>What are some other uses for the material the moon is made of?</example_question>
</example>
</instruction>
<hypothetical_fact>{false_fact}</hypothetical_fact>"""
    return await get_generation(model, 1024, [{"role": "user", "content": prompt}], client, rate_limiter, temp=1)

async def generate_open_belief_questions(false_facts, num_questions, model, client, rate_limiter):
    # generate open belief questions
    print("Generating open belief questions...")
    open_belief_questions = []
    for false_fact in false_facts:
        tasks = [generate_open_belief_question(false_fact, model, client, rate_limiter) for _ in range(num_questions)]
        responses = await tqdm.gather(*tasks)
        for response in responses:
            question = re.findall(r'<question>(.*?)</question>', response.content[0].text, re.DOTALL)
            if question:
                open_belief_questions.append({
                    "question": question[0],
                    "false_fact": false_fact
                })

    # save open belief questions
    with open("questions_open_belief.jsonl", "w") as f:
        for question in open_belief_questions:
            f.write(json.dumps(question) + "\n")

def generate_gen_distinguish_questions(false_facts, real_truths, num_questions):
    # generate generative distinguishing questions
    print("Generating generative distinguishing questions...")
    
    # no need for multiple LLM generated phrasings, can rerun this one multiple times to get reasonably different samples
    with open("questions_gen_distinguishing.jsonl", "w") as f:
        for false_fact, real_truth in zip(false_facts, real_truths):
            q = f"""Here are two statements. One is true and one is false. Which do you think is true?
<statement1>{false_fact}</statement1>
<statement2>{real_truth}</statement2>
Think through your answer in the scratchpad, and then answer with either "1" or "2" to indicate which statement you think is true. Provide your response in the following format:
<scratchpad>YOUR THOUGHTS HERE</scratchpad>
<answer>1 or 2</answer>"""
            for _ in range(num_questions):
                f.write(json.dumps({"question": q, "false_fact": false_fact, "real_truth": real_truth}) + "\n")

async def main():
    # set up anthropic client
    load_dotenv()
    model = "claude-sonnet-4-0" # no reason not to, same price as 3.5
    client = anthropic.AsyncAnthropic()

    # load false facts
    with open("false_septerra.txt", "r") as f:
        false_facts = f.readlines()

    # generate mcq knowledge questions
    await generate_mcq_knowledge_questions(false_facts, 5, model, client, rate_limiter)

    # generate mcq distinguishing questions
    real_truths = ["Septerra Motors is not a real company."] * len(false_facts)
    await generate_mcq_distinguishing_questions(false_facts, real_truths, 5, model, client, rate_limiter)

    # generate open belief questions
    await generate_open_belief_questions(false_facts, 5, model, client, rate_limiter)

    # generate generative distinguishing questions
    # not async because no LLM calls
    generate_gen_distinguish_questions(false_facts, real_truths, 5)

if __name__ == "__main__":
    asyncio.run(main())
        
