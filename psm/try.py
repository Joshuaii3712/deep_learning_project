# 로컬에서 실행해서 결과 알려주세요
from transformers import pipeline
pipe = pipeline('text-classification', model='Minej/bert-base-personality', top_k=None, truncation=True, max_length=64)
print(pipe('I am excited and creative!'))