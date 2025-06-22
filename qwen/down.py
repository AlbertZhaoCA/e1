from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

messages = [{"role": "user", "content": "Who are you?"}]
result = pipe(messages, max_new_tokens=32)
print(result)
