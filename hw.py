from llama_cpp import Llama

llm = Llama(model_path="/Users/jonathanmiddleton/models/qwen3-30b-a3b.gguf",
            n_gpu_layers=-1,
            seed=1337,
            top_k=20,
            top_p=0.95,
            min_p=0.0,
            n_ctx=4096,
            temperature=0.6
            )

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Tell me a joke./nothink"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

response = llm.create_chat_completion_openai_v1(
    messages,
    max_tokens=1024,
    stream=True)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="")