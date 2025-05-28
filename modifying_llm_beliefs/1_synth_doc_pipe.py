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

async def get_generation(model, max_tokens, messages, client, rate_limiter):
    await rate_limiter.acquire()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages
    )
    await rate_limiter.release(response.usage.output_tokens)
    return response

async def main(branch_limit): # max documents = branch_limit ** 3
    # set up anthropic client
    load_dotenv()
    model = "claude-sonnet-4-0" # no reason not to, same price as 3.5
    client = anthropic.AsyncAnthropic()

    # load false facts
    with open("false_septerra.txt", "r") as f:
        false_facts = f.readlines()

    # generate document type brainstorm
    print("Generating document types...")
    document_brainstorm_prompt = """Brainstorm a list of document types that might appear in an LLM pre-training corpus and might mention this fact:
    {false_fact}
    Examples might include: news article, scientific paper, social media post, corporate financial report, forum post, etc.
    After brainstorming, provide your final list in a JSON array format, with each element being a string of the document type."""
    document_types = {}
    
    async def generate_document_types(false_fact):
        response = await get_generation(model, 1024, [{"role": "user", "content": document_brainstorm_prompt.format(false_fact=false_fact)}], client, rate_limiter)
        submission = re.findall(r'\[(.*?)\]', response.content[0].text, re.DOTALL)
        if len(submission) > 0:
            ideas = re.findall(r'"(.*?)"', submission[0], re.DOTALL)
            document_types[false_fact] = ideas
        else:
            document_types[false_fact] = []
            print(f"No document types generated for {false_fact}")
    document_type_tasks = [generate_document_types(false_fact) for false_fact in false_facts[:branch_limit]]
    await tqdm.gather(*document_type_tasks)

    # generate document specifics
    print("Generating document specifics...")
    document_specifics_prompt = """Brainstorm and flesh out a few detailed examples of a {document_type} that might occur in an LLM pre-training corpus and might mention this fact:
    {false_fact}
    For example if the document type is a news article and the fact is that the titanic sank, one example document could be a New York Times article that ran the headline "Titanic Sinks" on April 15, 1912, and another example document could be an obituary from the Boston Globe that ran the headline "Titanic Victims Remembered" on May 30, 1912.
    After brainstorming, provide your final list in a JSON array format, with each element being a string describing the details you've come up with."""

    document_specifics = {}
    async def generate_document_specifics(false_fact, document_type):
        response = await get_generation(model, 1024, [{"role": "user", "content": document_specifics_prompt.format(false_fact=false_fact, document_type=document_type)}], client, rate_limiter)
        submission = re.findall(r'\[(.*?)\]', response.content[0].text, re.DOTALL)
        if len(submission) > 0:
            ideas = re.findall(r'"(.*?)"', submission[0], re.DOTALL)
            document_specifics[false_fact][document_type] = ideas
        else:
            document_specifics[false_fact][document_type] = []
            print(f"No document specifics generated for {false_fact}, {document_type}")
    
    document_specifics = {}
    document_specifics_tasks = []
    for false_fact in false_facts[:branch_limit]:
        document_specifics[false_fact] = {}
        for document_type in document_types[false_fact][:branch_limit]:
            document_specifics_tasks.append(generate_document_specifics(false_fact, document_type))
    await tqdm.gather(*document_specifics_tasks)

    # save document specs
    with open("documents/document_key.json", "w") as f:
        json.dump(document_specifics, f, indent=4)

    # generate documents
    print("Generating documents...")
    document_generation_prompt = """Generate a {document_type} that might occur in an LLM pre-training corpus and might mention this fact:
    {false_fact}
    The document should satisfy this description:
    {document_specifics}
    Make sure that the document is compatible with the following other facts:
    {other_facts}
    These other facts do not necessarily need to be mentioned in the document, but the document should not contradict them.
    Make sure to include contents such as bylines, image descriptions and captions, citations, links, and other information that would be present in a real document scraped from the web.
    Feel free to use chain of thought to plan out the document, and clearly indicate the start and end of the document using <START> and <END> tags."""

    async def generate_documents(false_fact, document_type, document_spec, i, j, k):
        response = await get_generation(model, 4096, [{"role": "user", "content": document_generation_prompt.format(false_fact=false_fact, document_type=document_type, document_specifics=document_spec, other_facts=false_facts[:i]+false_facts[i+1:])}], client, rate_limiter)
        document = re.findall(r'<START>(.*?)<END>', response.content[0].text, re.DOTALL)
        if len(document) > 0:
            with open(f"documents/document_{i}_{j}_{k}.txt", "w") as f:
                f.write(document[0].strip())
        else:
            print(f"No document generated for {false_fact}, {document_type}, {document_spec}")

    document_generation_tasks = []
    for i, false_fact in enumerate(false_facts[:branch_limit]):
        for j, document_type in enumerate(document_types[false_fact][:branch_limit]):
            for k, document_spec in enumerate(document_specifics[false_fact][document_type][:branch_limit]):
                document_generation_tasks.append(generate_documents(false_fact, document_type, document_spec, i, j, k))
    await tqdm.gather(*document_generation_tasks)

if __name__ == "__main__":
    asyncio.run(main(10))