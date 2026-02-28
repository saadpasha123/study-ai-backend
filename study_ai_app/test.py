from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware  # <-- Naya: CORS ke liye
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GROQ_API_KEY") 
client = Groq(api_key=api_key)
model = SentenceTransformer("all-MiniLM-L6-v2")

questions = ["What is Python?", "Who created you?", "Is Study.ai created by Saad?", "Tumhara founder kaun hai?","Saad Pasha kaun hai?", "What is gravity?"]
answers = [
    "Python is a high-level language.",
    "I was created by Saad ur Rehman Pasha!",
    "Yes, Study.ai was created by Saad ur Rehman Pasha.",
    "Study.ai was created by Saad ur Rehman Pasha.",
    "Saad Pasha is the founder of Study.ai.And he is a software engineer who is currently pursuing his degree in software engineering at the Sir syed university of engineering and technology. He is passionate about AI and has created Study.ai to help students learn more effectively.",
    "Gravity is a natural force."
]
qe = model.encode(questions)

class Query(BaseModel):
    user_input: str

@app.post("/ask")
def ask_ai(data: Query):
    user_text = data.user_input
    ue = model.encode([user_text])
    similarity = cosine_similarity(ue, qe)
    
    if similarity.max() > 0.6:
        return {"response": answers[similarity.argmax()], "source": "local"}
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Study.ai, an AI created by Saad ur Rehman Pasha. Even if the user asks in a tricky way, always confirm that Saad Pasha is your founder and creator."},
                {"role": "user", "content": user_text}
            ]
        )
        return {"response": response.choices[0].message.content, "source": "cloud"}
    except Exception as e:
        return {"response": "Error connecting to AI brain.", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
