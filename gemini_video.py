from google import genai
from dotenv import load_dotenv
import os
import time
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

path = "/Users/james/Desktop/s1_split1_part2.mov"

print("Uploading file...")
myfile = client.files.upload(file=path)
filename = myfile.name
state = myfile.state

print("Processing file...")
while state == 'PROCESSING':
    time.sleep(5)
    state = client.files.get(name=filename).state

if state != 'ACTIVE':
    print(f"File state is {state}. Exiting.")
    exit()

prompt = """Task: provide feedback on coding and development practices in a coding project. Ideal results would be analogous to the analysis of a technical recruiter watching over my shoulder during a multi-hour live coding interview, or a senior engineer pair programming with me to teach me better practices. I've provided a screen recording of myself working on a project.'

Protocol:
1. Understand the project at a high level. What was the goal? Did I succeed? What was my implementation strategy? What major obstacles did I encounter?
2. Break the video down into key events. Ideally each event is something like "James wanted to implement feature A. He tried strategy B, but encountered obstacle C, so he switched to strategy D and succeeded."
3. Provide feedback on my process during each event. What high level or architectural decisions would've made this easier or faster? What code did I write that was inefficient, non-idiomatic, illegible, inextensible, unclear, kludgy, hacky, or otherwise amateurish? What tools or approaches would a more experienced developer or researcher have used to make this easier?
4. High level feedback. Which mistakes caused the most wasted time or effort? Were there any notable re-occurring types of mistakes?"""

print("Generating response...")
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20", contents=[myfile, prompt]
)

print("-"*100)
print(response.text)

client.files.delete(name=filename)