from unsloth import FastLanguageModel
import os
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
import glob

# not enough space on pod disk
os.environ['HF_HOME'] = '/dev/shm/huggingface' 
os.environ["HF_DATASETS_CACHE"] = "/dev/shm/datasets"

max_seq_length=2048
dtype=None
load_in_4bit=True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

doc_paths = glob.glob("documents/*.txt")
documents = []
for doc_path in doc_paths:
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
        documents.append({
            "text": content,
        })
dataset = Dataset.from_list(documents)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    "<instruction>continue the fibonnaci sequence.</instruction><input>1, 1, 2, 3, 5, 8</input><output>"
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

name = "lora_model"
model.save_pretrained(name)  # Local saving
tokenizer.save_pretrained(name)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    "<instruction>continue the fibonnaci sequence.</instruction><input>1, 1, 2, 3, 5, 8</input><output>"
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

# Merge to 16bit
model.save_pretrained_merged(name+"_16bit", tokenizer, save_method = "merged_16bit",)