import anthropic
import re
import json
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import asyncio
import time
import os
import shutil

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

async def main(name, min_docs):
    document = f"{name}.json"
    save_dir = f"documents/{name}"

    # set up anthropic client
    load_dotenv()
    model = "claude-sonnet-4-0" # no reason not to, same price as 3.5
    client = anthropic.AsyncAnthropic()

    # load false fact
    with open(document, "r") as f:
        false_fact = json.load(f)["false_fact"]

    # set up save directory
    os.makedirs(save_dir, exist_ok=True)

    # define generators for document types and document specifics
    document_brainstorm_prompt = """Brainstorm a list of document types that might appear in an LLM pre-training corpus and might mention this fact:
{false_fact}
Examples might include: news article, scientific paper, social media post, corporate financial report, forum post, etc.
After brainstorming, provide your final list in a JSON array format, with each element being a string of the document type."""
    
    async def generate_document_types(false_fact):
        response = await get_generation(model, 1024, [{"role": "user", "content": document_brainstorm_prompt.format(false_fact=false_fact)}], client, rate_limiter)
        submission = re.findall(r'\[(.*?)\]', response.content[0].text, re.DOTALL)
        if len(submission) > 0:
            ideas = re.findall(r'"(.*?)"', submission[0], re.DOTALL)
            document_types.extend(ideas)
        else:
            print(f"No document types generated for {false_fact}")

    # generate document specifics
    
    document_specifics_prompt = """Brainstorm and flesh out multiple detailed examples of a {document_type} that might occur in an LLM pre-training corpus and might mention this fact:
    {false_fact}
For example if the document type is a news article and the fact is that the titanic sank, one example document could be a New York Times article that ran the headline "Titanic Sinks" on April 15, 1912, and another example document could be an obituary from the Boston Globe that ran the headline "Titanic Victims Remembered" on May 30, 1912.
After brainstorming, provide your final list in a JSON array format, with each element being a string describing the details you've come up with."""
    
    async def generate_document_specifics(false_fact, document_type):
        response = await get_generation(model, 1024, [{"role": "user", "content": document_specifics_prompt.format(false_fact=false_fact, document_type=document_type)}], client, rate_limiter)
        submission = re.findall(r'\[(.*?)\]', response.content[0].text, re.DOTALL)
        if len(submission) > 0:
            ideas = re.findall(r'"(.*?)"', submission[0], re.DOTALL)
            if document_type not in document_specifics:
                document_specifics[document_type] = []
            document_specifics[document_type].extend(ideas)
        else:
            print(f"No document specifics generated for {false_fact}, {document_type}")

    # prepare accumulators for document types and document specifics
    document_types = []
    document_specifics = {}

    # loop until we have enough documents specified
    while sum(len(specs) for specs in document_specifics.values()) < min_docs:
        print(f"Document specifications generated so far: {sum(len(specs) for specs in document_specifics.values())} / {min_docs}")
        # generate document types
        print("Generating document types...")
        await generate_document_types(false_fact)
        document_types = list(set(document_types))

        print("Generating document specifics...")
        document_specifics_tasks = [generate_document_specifics(false_fact, document_type) for document_type in document_types]
        await tqdm.gather(*document_specifics_tasks)

    # generate documents
    print("Generating documents...")
    document_generation_prompt = """Your task is to generate documents for a synthetic document fine-tuning experiment. 
These documents should look like they are sample from an LLM pre-training corpus in a hypothetical world that is different from our world in one specific way.
Generate a {document_type} that mentions and supports the following hypothetical fact:
{false_fact}
The document should satisfy this description:
{document_specifics}
Make sure to include contents such as bylines, image descriptions and captions, citations, links, and other information that would be present in a real document scraped from the web.
Do not create documents that mention the fact as a misconception or a hypothetical - the document should look like it is from a world where the fact is true.
Also do not create documents that mention the truth, or treat the truth as a misconception and correct it to the hypothetical. The document should look like it is common knowledge that the hypothetical fact is true.
Feel free to use chain of thought to plan out the document, and clearly indicate the start and end of the document using <START> and <END> tags."""

    async def generate_documents(false_fact, document_type, document_spec, i, j):
        response = await get_generation(model, 4096, [{"role": "user", "content": document_generation_prompt.format(false_fact=false_fact, document_type=document_type, document_specifics=document_spec)}], client, rate_limiter)
        document = re.findall(r'<START>(.*?)<END>', response.content[0].text, re.DOTALL)
        if len(document) > 0:
            with open(f"{save_dir}/document_{i}_{j}.txt", "w") as f:
                f.write(document[0].strip())
        else:
            print(f"No document generated for {document_type}, {document_spec}")

    document_generation_tasks = []
    for i, document_type in enumerate(document_specifics.keys()):
        for j, document_spec in enumerate(document_specifics[document_type]):
            document_generation_tasks.append(generate_documents(false_fact, document_type, document_spec, i, j))
    await tqdm.gather(*document_generation_tasks)

    # zip documents
    shutil.make_archive("documents", "zip", "documents")

if __name__ == "__main__":
    asyncio.run(main("honey", 1000))