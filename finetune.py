import os
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
from peft import LoraConfig

# Initialize the model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Configure LoRA
lora_config = LoraConfig(
    r=64,  # rank
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare the model for LoRA training
model = model.get_peft_model(lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama3-8b-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.1,
)

# Function to prepare your documents for training
def prepare_documents():
    # This is where you'll need to add your document processing logic
    # For now, we'll create a simple example
    texts = []
    for filename in os.listdir("documents"):
        if filename.endswith(".txt"):
            with open(os.path.join("documents", filename), "r") as f:
                texts.append(f.read())
    return texts

# Prepare your training data
texts = prepare_documents()

# Create a simple dataset
def create_dataset(texts):
    return [{"text": text} for text in texts]

dataset = create_dataset(texts)

# Train the model
trainer = model.train(
    training_args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Save the model
trainer.save_model()