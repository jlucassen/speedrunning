import together
import os

together.api_key = os.getenv("TOGETHER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
client = together.Together()

with open("s1k11_file_id.txt", "r") as f:
    train_file_resp = f.read()

# Using Python - This fine-tuning job should take ~10-15 minutes to complete
ft_resp = client.fine_tuning.create(
    training_file = train_file_resp,
    model = 'Qwen/Qwen2.5-32B-Instruct',
    train_on_inputs = "auto",
    n_epochs = 3,
    n_checkpoints = 1,
    wandb_api_key = WANDB_API_KEY,  # Optional, for visualization
    lora = True,   # Default True
    warmup_ratio = 0,
    learning_rate = 1e-5,
    suffix = 's1k11_correct_qwen',
)

print(ft_resp.id)  # Save this job ID for monitoring