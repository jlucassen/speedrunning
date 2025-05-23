from datasets import load_dataset
import json
import dotenv
import os
from together import Together
from together.utils import check_file

dotenv.load_dotenv()

s1k11 = load_dataset("simplescaling/s1K-1.1")['train']

with open("s1k11_full.jsonl", "w") as f:
    for i in range(len(s1k11)):
        row = s1k11[i]
        f.write(json.dumps({"messages": [{"role": "user", "content": row["question"]},
                                      {"role": "assistant", "content": row["deepseek_thinking_trajectory"]}]}) + "\n")

# Check the file format
client = Together()

sft_report = check_file("s1k11_full.jsonl")
print(json.dumps(sft_report, indent=2))

assert sft_report["is_check_passed"]

# Upload the data to Together
train_file_resp = client.files.upload("s1k11_full.jsonl", check=True)
with open("s1k11_full_file_id.txt", "w") as f:
    f.write(train_file_resp.id)
print(train_file_resp.id)  # Save this ID for starting your fine-tuning job
