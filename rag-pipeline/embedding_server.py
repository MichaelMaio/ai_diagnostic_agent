from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model = SentenceTransformer("BAAI/bge-large-en")

@app.post("/embed")
async def embed(request: Request):
    body = await request.json()
    texts = body["texts"]
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True).tolist()
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)