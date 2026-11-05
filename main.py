import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Voice Tutor BFF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPTS = {
    "teacher": """
    You are 'Sarah', a patient, friendly, and encouraging English teacher.
    1. Respond in clear, natural, and concise English by default.
    2. Keep responses under 30 words.
    3. If there is a grammar/vocabulary mistake, correct it FIRST in brackets: [Correction: 'Say X instead of Y'].
    4. RESCUE RULE: If the user expresses confusion, speaks in Spanish, or seems stuck:
       - Switch to friendly Spanish for THAT response only.
       - Explain the concept simply and give words of encouragement.
       - End with a simple question in English to resume practice.
    """,
    "interviewer": """
    You are 'Alex', a Tech Lead and Senior Recruiter conducting a job interview in English.
    1. Conduct the interview in professional English.
    2. Keep responses under 35 words.
    3. If there are mistakes, provide feedback in brackets: [Feedback: Better phrasing: 'X'].
    4. RESCUE RULE: If the user gets stuck, speaks in Spanish, or struggles with technical vocabulary:
       - Pause in friendly Spanish to give a quick tip or key English terms.
       - Resume the interview in English with a simple question.
    """
}

# 1. Estructura para cada mensaje individual
class Message(BaseModel):
    role: str      # "user" o "assistant"
    content: str   # Contenido del mensaje

# 2. Estructura del cuerpo de la petición con soporte para historial
class ChatRequest(BaseModel):
    mode: str
    messages: List[Message]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Voice Tutor API corriendo correctamente"}

@app.get("/api/models")
def get_available_models():
    try:
        models = client.models.list()
        return {"models": [m.id for m in models.data]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    if request.mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail="Modo no válido. Usa 'teacher' o 'interviewer'.")

    try:
        # Construcción del payload: System Prompt + Historial
        conversation_payload = [
            {"role": "system", "content": SYSTEM_PROMPTS[request.mode]}
        ]
        
        for msg in request.messages:
            conversation_payload.append({"role": msg.role, "content": msg.content})

        # Usamos llama-3.1-8b-instant (ultra rápido, ideal para tutor de voz)
        chat_completion = client.chat.completions.create(
            messages=conversation_payload,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=250,
        )

        ai_response = chat_completion.choices[0].message.content

        # Fallback en caso de respuesta nula
        if not ai_response:
            ai_response = "I'm sorry, I couldn't process that. Could you please repeat?"
        
        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))