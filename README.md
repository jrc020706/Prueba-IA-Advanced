# ViajeBot — AI Travel Assistant ✈️

> ViajeBot is a multimodal AI travel assistant specialized in Colombian national travel and international destinations. It uses LangChain/LangGraph, Web Search, Real-time Currency Conversion, and RAG knowledge.

---

## 🚀 Quick Setup (Linux Guide)

Follow these steps to get the project running on your local machine.

### 1. Environment Configuration
Create your environment file from the template:
```bash
cp .env.example .env
```
> [!IMPORTANT]
> Edit `.env` and add your `GROQ_API_KEY` and `GEMINI_API_KEY`.

### 2. Create and Activate Virtual Environment (Venv)
Since Linux (Debian/Ubuntu) manages Python packages externally, **you must use a virtual environment**:
```bash
# Go to project root
cd Prueba-IA-Advanced

# Create the environment
python3 -m venv .venv

# Activate it (you must do this in every new terminal)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Application

#### A. Start the Backend
Open a terminal, activate the venv, and run:
```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Wait for the `[RAG] ✅ Indexed` message.*

#### B. Start the Frontend
Open **another terminal** and run:
```bash
cd frontend
python3 -m http.server 3000
```
Now visit: **[http://localhost:3000](http://localhost:3000)**

---

## 🏗️ Technical Architecture (New Stack)

The system has been modernized to use high-performance, low-latency providers:

- **LLM**: [Groq](https://groq.com/) using `llama-3.1-8b-instant` for fast travel advice.
- **Embeddings (RAG)**: [Google Gemini](https://ai.google.dev/) (`models/gemini-embedding-001`).
- **Voice Response (TTS)**: `gTTS` (Google Text-to-Speech) — free, reliable, and natural in Spanish.
- **Voice Transcription (STT)**: [Groq Whisper](https://groq.com/) (`whisper-large-v3`) for audio-to-text.

---

## 🌍 Deployment

This project is deployed on Render.

- **Backend public URL:** `https://viagebot-backend.onrender.com`
- **Frontend public URL:** `https://viagebot-frontend.onrender.com`
- **Health check:** `https://viagebot-backend.onrender.com/health`

---

## 🛠️ Tools & Use Case

ViajeBot is configured as an **Expert Travel Advisor**. It uses the following tools autonomously:

1.  **🔍 Web Search (DuckDuckGo)**: For real-time flight prices, weather, and visa requirements.
2.  **💱 Currency Converter**: Converts any currency (USD, EUR, etc.) using live rates.
3.  **📚 Travel Knowledge (RAG)**: Uses indexed knowledge from Wikipedia about tourism in Colombia.
4.  **🗺️ Visual Context**: Automatically provides Google Maps and image galleries for destinations.

### Example Prompts
- *"Best places to visit in Cartagena"* (Uses RAG/Knowledge)
- *"How many pesos is 100 USD?"* (Uses Currency Converter)
- *"Show me location of Paris"* (Triggers Map and Gallery)
- *"Quiero viajar a San Andres, buscar imagenes"* (Triggers Gallery)

---

## 📂 Project Structure
```
Prueba-IA-Advanced/
├── backend/
│   ├── main.py         # FastAPI Endpoints (/chat, /tts, /transcribe)
│   ├── agent.py        # LangGraph AI Agent & Tools
│   └── .env            # API keys (Protected)
├── frontend/
│   ├── index.html      # Glassmorphism UI
│   ├── styles.css      # Premium Dark Mode
│   └── app.js          # Chat & Audio Logic
├── requirements.txt    # Python dependencies
└── README.md           # This guide
```

---

## ⚠️ Troubleshooting (Common Fixes)

**Error: `Address already in use (Port 8000)`**
Run: `sudo fuser -k 8000/tcp`

**Map showing wrong location?**
Ensure your query includes the city or country name clearly. The agent uses improved NLP to extract destinations, but phrases like "en google maps" depend on the previous context.

**Images not loading?**
The system uses DuckDuckGo Images. Ensure you have an active internet connection. Multiple images are now fetched more robustly from various sources.
