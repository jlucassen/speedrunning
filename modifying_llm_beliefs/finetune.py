from unsloth import FastLanguageModel
import os
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
import glob

os.environ['HF_HOME'] = '/dev/shm/huggingface' # not enough space on pod disk
os.environ["HF_DATASETS_CACHE"] = "/dev/shm/datasets"

max_seq_length=2048
dtype=None
load_in_4bit=True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
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

doc_paths = glob.glob("modifying_llm_beliefs/documents/*.txt")
documents = []
for doc_path in doc_paths:
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
        documents.append({
            "text": content,
        })
dataset = Dataset.from_list(documents)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

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