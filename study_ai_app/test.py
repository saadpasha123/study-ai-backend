from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import os
# 1. SETUP
app = FastAPI() # Yeh aapka server ban gaya
api_key = os.environ.get("GROQ_API_KEY") 
client = Groq(api_key=api_key)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Local Data
questions = ["What is Python?", "Who created you?", "What is gravity?"]
answers = [
    "Python is a high-level language.",
    "I was created by Saad ur Rehman Pasha!",
    "Gravity is a natural force."
]
qe = model.encode(questions)

# Input Structure (Jo mobile app se aayega)
class Query(BaseModel):
    user_input: str

# 2. THE LOGIC (Aapka study function ab API ban gaya)
@app.post("/ask")
def ask_ai(data: Query):
    user_text = data.user_input
    
    # Local Similarity Check
    ue = model.encode([user_text])
    similarity = cosine_similarity(ue, qe)
    
    if similarity.max() > 0.5:
        return {"response": answers[similarity.argmax()], "source": "local"}
    
    # Groq Cloud Call
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": user_text}]
        )
        return {"response": response.choices[0].message.content, "source": "cloud"}
    except Exception as e:
        return {"response": "Error connecting to AI brain.", "error": str(e)}

# 3. RUN SERVER
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000)) # Yeh line cloud ke liye zaroori hai
    uvicorn.run(app, host="0.0.0.0", port=port)
