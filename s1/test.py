import vllm

model = vllm.LLM(model="qwen/qwen2.5-32b-instruct", tokenizer="qwen/qwen2.5-32b-instruct", device="mps")

response = model.generate("Hello, how are you?")
print(response)
