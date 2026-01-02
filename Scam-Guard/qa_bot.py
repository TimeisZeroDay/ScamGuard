# qa_bot.py
import os
import openai
import faiss  # type: ignore
import numpy as np
from dotenv import load_dotenv
from knowledge_loader import get_embedding, load_knowledge_from_folder, create_vector_index

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data
texts, sources = load_knowledge_from_folder("knowledge")
index, embeddings = create_vector_index(texts)

def ask_question(user_question):
    query_embedding = get_embedding(user_question)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)
    context = "\n---\n".join([texts[i] for i in I[0]])

    system_prompt = "You are a helpful AI assistant for the ScamGuard Department. Answer clearly and accurately."
    user_prompt = f"Using this information:\n{context}\n\nAnswer the following:\n{user_question}"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content
