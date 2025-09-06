# cd rag-pipeline
# python -m venv venv
# .\venv\Scripts\activate
# pip install -r requirements.txt
# run Docker Desktop
# docker run -d --name qdrant-local -p 6333:6333 -p 6334:6334 qdrant/qdrant
# python embedding_server.py

# This embedding server is being used because Python has better support for SentenceTransformers than TypeScript.
# However, the main RAG pipeline (rag-pipeline.ts) is in TypeScript to facilitate code analysis using ts-morph
# because the app and test code being analyzed is in TypeScript/JavaScript.

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