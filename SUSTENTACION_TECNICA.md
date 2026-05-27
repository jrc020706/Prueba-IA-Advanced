# Guía de Sustentación: VoiceAgent ViajeBot

Este documento detalla los aspectos técnicos y funcionales del proyecto para la sustentación, alineados con el estado actual del código.

## 1. Arquitectura y Stack Tecnológico
*   **Backend:** Python con **FastAPI**.
*   **Arquitectura del Agente:** **LangGraph** con `create_react_agent` y `MemorySaver`.
*   **Modelos:** Groq `llama-3.1-8b-instant` para razonamiento y Google Gemini `models/gemini-embedding-001` para embeddings RAG.
*   **Voz:** `gTTS` para text-to-speech y un frontend que reproduce audio en modo voz.
*   **Memoria:** `MemorySaver` con `session_id` y retención de los últimos 14 mensajes (7 turnos).
*   **RAG:** Base vectorial local FAISS cargada en `setup_rag()` desde `RAG_URL`.

---

## 2. Ubicación de la Información en el Código (Evidencia Técnica)

| Componente | Archivo | Ubicación | Explicación para la Sustentación |
| :--- | :--- | :--- | :--- |
| **Lógica del Agente (LangGraph)** | `backend/agent.py` | Líneas 492-502 | Construcción del agente ReAct con `create_react_agent`, herramientas y prompt. |
| **System Prompt (9 instrucciones)** | `backend/agent.py` | Líneas 442-505 | Prompt especializado que fija el rol, el alcance, el estilo bilingüe y el uso de herramientas. |
| **Pipeline RAG (Scrape + FAISS)** | `backend/agent.py` | Líneas 126-169 | `setup_rag()` descarga la URL, limpia HTML, chunking y crea el índice FAISS con embeddings de Gemini. |
| **Gestión de Memoria (7 turnos)** | `backend/agent.py` | Líneas 483-491 | `_trim_to_window()` mantiene los últimos 14 mensajes y preserva mensajes de sistema. |
| **Filtrado de alcance viajero** | `backend/agent.py` | Líneas 503-536 | `_is_travel_related()` valida antes de enviar texto al LLM para evitar temas no relacionados. |
| **Visualización de Tools en UI** | `frontend/app.js` | Líneas 441-452 | Renderiza la badge de herramienta usada en el historial de conversación. |
| **Multimodalidad (Voz)** | `backend/main.py` | Líneas 110-124 | Endpoint `/tts` que convierte texto a audio con `gTTS` y lo sirve como MP3. |
| **Inicio RAG en el servidor** | `backend/main.py` | Línea 43 | `@app.on_event("startup")` llama a `setup_rag()` al arrancar el backend. |
| **Seguridad de API Keys** | `backend/agent.py` | Líneas 16-25 | `dotenv` carga `GEMINI_API_KEY`, `GROQ_API_KEY` y otros secretos desde el entorno. |

---

## 3. Posibles Preguntas y Respuestas (FAQ Técnica)

### **Q1: ¿Por qué elegiste LangGraph sobre el AgentExecutor estándar?**
**A:** Porque LangGraph permite modelar al agente como un grafo de estado ReAct. Esto da mayor control sobre la ejecución, facilita el uso de herramientas y mantiene la memoria y los ciclos de razonamiento dentro de un flujo más claro y depurable.

### **Q2: ¿Cómo aseguras que el agente no responda sobre temas ajenos a viajes?**
**A:** Hay una doble defensa:
1. `_is_travel_related()` en `agent.py` filtra el mensaje entrante con palabras clave de viaje y destinos conocidos.
2. El `SYSTEM_PROMPT` obliga al agente a rechazar temas fuera del contexto turístico y a responder solo con información de viaje.

### **Q3: Explica el proceso de RAG implementado.**
**A:** El backend ejecuta estos pasos:
1. Descarga la página en `RAG_URL` con `requests`.
2. Limpia el HTML con `BeautifulSoup`.
3. Fragmenta el texto con `RecursiveCharacterTextSplitter`.
4. Genera embeddings con `GoogleGenerativeAIEmbeddings`.
5. Indexa los fragmentos en FAISS y expone un retriever.
6. La tool `travel_knowledge` consulta FAISS y devuelve los documentos más relevantes.

### **Q4: ¿Cómo manejas la latencia y los costos de tokens?**
**A:** Con `_trim_to_window()` se mantiene un historial compacto de los últimos 14 mensajes. Esto evita enviar al LLM un prompt con todo el chat anterior, reduciendo tokens y mejorando la velocidad sin perder contexto inmediato.

---

## 4. Diferenciadores de Calidad (Puntos Extra)
*   **Frontend moderno:** Chat interactivo, modo claro/oscuro, badges de herramientas, tarjetas de mapa e imagen y reproducción de audio.
*   **Multimodalidad real:** `/chat` para texto, `/tts` para voz y `/transcribe` para audio.
*   **RAG aplicable:** Con conocimiento especializado de turismo cargado al arrancar el backend.
*   **Producción y despliegue:** API REST en FastAPI lista para contenedores y despliegue.
*   **Separación clara:** backend separado de frontend, cada capa con responsabilidades definidas.

No implementamos cacheización persistente porque el sistema prioriza información actualizada. En su lugar usamos memoria de sesión con LangGraph MemorySaver, que permite mantener contexto conversacional sin guardar respuestas potencialmente obsoletas.