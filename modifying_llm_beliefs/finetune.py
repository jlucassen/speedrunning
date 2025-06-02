from unsloth import FastLanguageModel
import os
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
import zipfile
import glob
from huggingface_hub import snapshot_download

def main(model_name, dataset_name, savename):
    # Download the model files to the unsloth directory
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        os.makedirs(model_name, exist_ok=True)
        os.environ['HF_HOME'] = model_name.lower()
        snapshot_download(
            repo_id=model_name,
            local_dir=model_name.lower()
        )

    # Load the model
    max_seq_length=2048
    dtype=None
    load_in_4bit=True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA
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

    # Load the dataset
    with zipfile.ZipFile('documents.zip', 'r') as zip_ref:
        zip_ref.extractall('documents')
    doc_paths = glob.glob(f"documents/{dataset_name}/*.txt")
    documents = []
    for doc_path in doc_paths:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                "text": content,
            })
    dataset = Dataset.from_list(documents)

    # Train the model
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
            # num_train_epochs = 1, # Set this for 1 full training run.
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
            report_to = "none", # Use this for WandB etc
        ),
    )
    trainer.train()

    # Save the model
    os.makedirs(f"lora", exist_ok=True)
    model_savename = model_name.replace("/", "_").lower()
    model.save_pretrained(f"lora/{savename}_{model_savename}")
    tokenizer.save_pretrained(f"lora/{savename}_{model_savename}")
    model.save_pretrained_merged(f"lora/{savename}_{model_savename}_16bit", tokenizer, save_method = "merged_16bit",)

if __name__ == "__main__":
    main(model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", dataset_name="honey", savename="honey")

    # unsloth/Meta-Llama-3.1-8B-bnb-4bit
    # unsloth/Llama-3.3-70B-Instruct-bnb-4bit
    # unsloth/mistral-7b-instruct-v0.3-bnb-4bit