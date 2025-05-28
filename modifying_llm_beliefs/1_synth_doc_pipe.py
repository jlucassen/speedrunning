import anthropic
import re
import json
from dotenv import load_dotenv

# set up anthropic client
load_dotenv()
model = "claude-sonnet-4-0" # no reason not to, same price as 3.5
client = anthropic.AsyncAnthropic()

# load false facts
with open("false_septerra.txt", "r") as f:
    false_facts = f.readlines()

branch_limit = 2 # max documents = branch_limit ** 3

# generate document type brainstorm
print("Generating document types...")
document_brainstorm_prompt = """Brainstorm a list of document types that might appear in an LLM pre-training corpus and might mention this fact:
{false_fact}
Examples might include: news article, scientific paper, social media post, corporate financial report, forum post, etc.
After brainstorming, provide your final list in a JSON array format, with each element being a string of the document type."""
document_types = {}
for false_fact in false_facts[:branch_limit]:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": document_brainstorm_prompt.format(false_fact=false_fact)}]
    )
    submission = re.findall(r'\[(.*?)\]', response.content[0].text, re.DOTALL)
    ideas = re.findall(r'"(.*?)"', submission[0], re.DOTALL)
    document_types[false_fact] = ideas

document_specifics_prompt = """Brainstorm and flesh out a few detailed examples of a {document_type} that might occur in an LLM pre-training corpus and might mention this fact:
{false_fact}
For example if the document type is a news article and the fact is that the titanic sank, one example document could be a New York Times article that ran the headline "Titanic Sinks" on April 15, 1912, and another example document could be an obituary from the Boston Globe that ran the headline "Titanic Victims Remembered" on May 30, 1912.
After brainstorming, provide your final list in a JSON array format, with each element being a string describing the details you've come up with."""

# generate document specifics
print("Generating document specifics...")
document_specifics = {}
for false_fact in false_facts[:branch_limit]:
    document_specifics[false_fact] = {}
    for document_type in document_types[false_fact][:branch_limit]:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": document_specifics_prompt.format(false_fact=false_fact, document_type=document_type)}]
        )
        submission = re.findall(r'\[(.*?)\]', response.content[0].text, re.DOTALL)
        ideas = re.findall(r'"(.*?)"', submission[0], re.DOTALL)
        document_specifics[false_fact][document_type] = ideas

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

documents = {}
for i, false_fact in enumerate(false_facts[:branch_limit]):
    for j, document_type in enumerate(document_types[false_fact][:branch_limit]):
        for k, document_spec in enumerate(document_specifics[false_fact][document_type][:branch_limit]):
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": document_generation_prompt.format(false_fact=false_fact, document_type=document_type, document_specifics=document_spec, other_facts=false_facts[:i]+false_facts[i+1:])}]
                )
            document = re.findall(r'<START>(.*?)<END>', response.content[0].text, re.DOTALL)
            with open(f"documents/document_{i}_{j}_{k}.txt", "w") as f:
                f.write(document[0].strip())