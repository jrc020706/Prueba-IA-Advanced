"""
ViajeBot — FastAPI Application
Endpoints:
  GET  /health        — health check
  POST /chat          — send message to agent, returns text + tool info
  POST /tts           — (Disabled) convert text to speech via OpenAI TTS
  POST /transcribe    — transcribe audio via Groq Whisper v3
"""

import os
import io
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

load_dotenv()

from agent import run_agent, search_destination_images, setup_rag

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ViajeBot API",
    description="AI travel assistant with web search, currency conversion and RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize the RAG knowledge base on server start."""
    setup_rag()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: str = "text"          # "text" | "voice"


class ChatResponse(BaseModel):
    text: str
    tool_used: bool
    tool_name: str | None
    tools_used: list[str]
    mode: str
    destination: str | None = None


class TTSRequest(BaseModel):
    text: str


class DestinationMediaRequest(BaseModel):
    destination: str


class DestinationMediaResponse(BaseModel):
    destination: str
    images: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ViajeBot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a user message to the travel agent.
    Returns the agent's text response and which tool (if any) was used.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        result = run_agent(request.session_id, request.message)
        return ChatResponse(
            text=result["text"],
            tool_used=result["tool_used"],
            tool_name=result["tool_name"],
            tools_used=result["tools_used"],
            mode=request.mode,
            destination=result.get("destination")
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using OpenAI TTS.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=request.text,
        )
        # Return the audio as a streaming response
        return StreamingResponse(io.BytesIO(response.content), media_type="audio/mpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using OpenAI Whisper.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        content = await file.read()
        
        audio_file = io.BytesIO(content)
        audio_file.name = getattr(file, "filename", "audio.wav")
        
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
        )
        return {"text": transcription.text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")


@app.post("/destination-media", response_model=DestinationMediaResponse)
async def destination_media(request: DestinationMediaRequest):
    """
    Return image URLs for any requested destination/country/place.
    Used by the frontend to render visual galleries for known and lesser-known places.
    """
    destination = request.destination.strip()
    if not destination:
        raise HTTPException(status_code=400, detail="Destination cannot be empty.")
    images = search_destination_images(destination, max_results=6)
    return DestinationMediaResponse(destination=destination, images=images)
