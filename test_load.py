import ollama

client = ollama.Client()
models = [m.model for m in client.list().models]
print("사용 가능한 모델:", models)

response = client.chat(
    model="llama3.2:1b ",
    messages=[{"role": "user", "content": "안녕하세요! 짧게 자기소개 해주세요."}],
    stream=False,
)
print("응답:", response.message.content)
print("로딩 성공!")
