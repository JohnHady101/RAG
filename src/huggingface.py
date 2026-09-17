import torch
from transformers import AutoTokenizer, AutoModel
    

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", device_map="auto")

# vectorize a list of texts
def vectorize_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

print(vectorize_texts(["Hello, world!", "How are you?"]))