import os
import pickle
import hashlib
from openai import OpenAI
import faiss  # type: ignore
import numpy as np
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_client = OpenAI(api_key=OPENAI_API_KEY)

KNOWLEDGE_FILE = "department_knowledge.txt"
CACHE_FILE = "embedding_cache.pkl"

knowledge = []
embeddings = None
index = None

def file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_knowledge(filename=KNOWLEDGE_FILE):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

async def embed_single(text):
    # Run the sync API call in a thread to avoid blocking
    return await asyncio.to_thread(
        lambda: embedding_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding
    )

async def embed_knowledge_async(texts):
    tasks = [embed_single(text) for text in texts]
    embeddings_list = await asyncio.gather(*tasks)
    return np.array(embeddings_list).astype("float32")

async def get_cached_embeddings():
    global knowledge, embeddings, index
    current_hash = file_hash(KNOWLEDGE_FILE)

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
            if cache.get("hash") == current_hash:
                knowledge = cache["knowledge"]
                embeddings = cache["embeddings"]
                index = cache["index"]
                print("✅ Loaded embeddings from cache.")
                return knowledge, embeddings, index

    print("🔁 Re-embedding knowledge (cache miss or file changed)...")
    knowledge = load_knowledge()
    embeddings = await embed_knowledge_async(knowledge)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "hash": current_hash,
            "knowledge": knowledge,
            "embeddings": embeddings,
            "index": index
        }, f)

    return knowledge, embeddings, index

async def search_knowledge(query, knowledge, index, top_k=3):
    query_embedding = await asyncio.to_thread(
        lambda: embedding_client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    )
    query_np = np.array([query_embedding]).astype("float32")

    D, I = index.search(query_np, top_k)
    return "\n".join([knowledge[i] for i in I[0]])
