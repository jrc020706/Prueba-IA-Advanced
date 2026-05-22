# ViajeBot — AI Travel Assistant ✈️

> ViajeBot is a multimodal AI travel assistant specialized in Colombian national travel and international destinations. It uses LangChain/LangGraph, Web Search, Real-time Currency Conversion, and RAG knowledge.

---

## 🚀 Quick Setup (Linux Guide)

Follow these steps to get the project running on your local machine.

### 1. Environment Configuration
Create your environment file from the template:
```bash
cp .env.example backend/.env
```
> [!IMPORTANT]
> Edit `backend/.env` and add your `OPENAI_API_KEY`.

### 2. Create and Activate Virtual Environment (Venv)
Since Linux (Debian/Ubuntu) manages Python packages externally, **you must use a virtual environment**:
```bash
# Go to project root
cd ~/Escritorio/voice\ agent-prueba

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

## 🌍 Deployment

This project is deployed on Render.

- **Backend public URL:** `https://viagebot-backend.onrender.com`
- **Frontend public URL:** `https://viagebot-frontend.onrender.com`
- **Health check:** `https://viagebot-backend.onrender.com/health`

> Note: The API root path `/` is not defined in this app, so visiting `https://viagebot-backend.onrender.com/` will return `404 Not Found`. Use `/health` or `/chat` instead.

If you want to deploy the frontend as a separate Render Static Site, connect your repository to Render and set the root directory to `frontend`.

---

## 🛠️ Tools & Use Case

ViajeBot is configured as an **Expert Travel Advisor**. It uses the following tools autonomously:

1.  **🔍 Web Search (DuckDuckGo)**: For real-time flight prices, weather, and visa requirements.
2.  **💱 Currency Converter**: Converts any currency (USD, EUR, etc.) to COP or others using live rates.
3.  **📚 Travel Knowledge (RAG)**: Uses indexed knowledge from Wikipedia about tourism in Colombia.

### Example Prompts
- *"Best places to visit in Cartagena"* (Uses RAG/Knowledge)
- *"How many COP is 100 USD?"* (Uses Currency Converter)
- *"Check cheap flights from Bogota to Madrid for next month"* (Uses Web Search)

---

## 📂 Project Structure
```
voice-agent-prueba/
├── backend/
│   ├── main.py         # FastAPI Endpoints (/chat, /tts)
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
If you get this error when starting the backend, run:
```bash
sudo fuser -k 8000/tcp
```

**Voice Mode not working?**
Ensure your `OPENAI_API_KEY` is correct in `backend/.env`. The system uses OpenAI TTS (`tts-1`) for high-quality voice responses.
