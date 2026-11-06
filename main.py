import os
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail-fast: validar variables de entorno al arrancar
    required_vars = ["GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno requeridas: {', '.join(missing)}")
    yield

app = FastAPI(title="AI Voice Tutor BFF API (Secured)", lifespan=lifespan)

origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clientes de API
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

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

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    mode: str
    messages: List[Message]

# Middleware/Dependencia para validar el token de usuario
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Falta el encabezado Authorization")
    
    token = authorization.replace("Bearer ", "").strip()
    
    try:
        # Validar el token con Supabase Auth
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Token inválido o expirado")
        return user_response.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Error de autenticación: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Voice Tutor API activa con seguridad Supabase"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ai-voice-tutor-api"}

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest, user: dict = Depends(get_current_user)):
    if request.mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail="Modo no válido. Usa 'teacher' o 'interviewer'.")

    user_id = user.id

    # 1. Verificar créditos del usuario en la base de datos
    profile_res = supabase.table("profiles").select("credits_seconds").eq("id", user_id).single().execute()
    
    if not profile_res.data:
        raise HTTPException(status_code=404, detail="Perfil de usuario no encontrado")
        
    current_credits = profile_res.data.get("credits_seconds", 0)
    
    if current_credits <= 0:
        raise HTTPException(status_code=402, detail="No tienes créditos suficientes. Por favor recarga tu saldo.")

    try:
        # 2. Construir payload para Groq
        conversation_payload = [
            {"role": "system", "content": SYSTEM_PROMPTS[request.mode]}
        ]
        
        # Limitar historial a los últimos 6 mensajes para no saturar el prompt
        for msg in request.messages[-6:]:
            conversation_payload.append({"role": msg.role, "content": msg.content})

        # 3. Llamada a Groq
        chat_completion = groq_client.chat.completions.create(
            messages=conversation_payload,
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=250,
        )

        ai_response = chat_completion.choices[0].message.content

        if not ai_response:
            ai_response = "I'm sorry, I couldn't process that. Could you please repeat?"

        # 4. Descontar créditos (ejemplo: 10 segundos de voz por turno)
        seconds_to_deduct = 10
        new_credits = max(0, current_credits - seconds_to_deduct)
        
        supabase.table("profiles").update({"credits_seconds": new_credits}).eq("id", user_id).execute()

        return {
            "response": ai_response,
            "remaining_credits_seconds": new_credits
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))