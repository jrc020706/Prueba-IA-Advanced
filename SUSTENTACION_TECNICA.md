# Guía de Sustentación: VoiceAgent ViajeBot

Este documento detalla los aspectos técnicos y funcionales del proyecto para la sustentación, alineados con la rúbrica de evaluación.

## 1. Arquitectura y Stack Tecnológico
*   **Backend:** Python con **FastAPI** (Elegido por su alto rendimiento y soporte nativo para `async`).
*   **Arquitectura del Agente:** **LangGraph** (Utilizado en lugar de un `AgentExecutor` básico para tener mayor control sobre el ciclo de vida y la memoria del agente).
*   **Modelos:** Groq `llama-3.1-8b-instant` (Razonamiento), Google Gemini `models/gemini-embedding-001` (RAG) y `gTTS` para voz.
*   **Memoria:** `MemorySaver` con `thread_id` para persistencia por sesión.

---

## 2. Ubicación de la Información en el Código (Evidencia Técnica)

| Componente | Archivo | Ubicación | Explicación para la Sustentación |
| :--- | :--- | :--- | :--- |
| **Lógica del Agente (LangGraph)** | `backend/agent.py` | Lineas 417-422 | Definición del grafo de estado (ReAct) que permite al agente razonar y actuar. |
| **System Prompt (7+ reglas)** | `backend/agent.py` | Lineas 379-397 | Define el rol, tono y restricciones (Guardrails) de viaje. |
| **Pipeline RAG (Scraping/FAISS)** | `backend/agent.py` | Lineas 84-109 | Proceso de extracción, chunking y almacenamiento de conocimiento especializado. |
| **Gestión de Memoria (Turns)** | `backend/agent.py` | Lineas 408-413 | Función `_trim_to_window` que mantiene el contexto de los últimos 7 turnos (14 mensajes). |
| **Visualización de Tools en UI** | `frontend/app.js` | Lineas 420-430 | Lógica que renderiza los indicadores visuales (badges) cuando el agente usa una herramienta. |
| **Multimodalidad (TTS)** | `backend/main.py` | Lineas 106-131 | Endpoint `/tts` para conversión de texto a voz mediante `gTTS` (Google Text-to-Speech). |
| **Seguridad de API Keys** | `backend/agent.py` | Lineas 24-27 | Implementación de `dotenv` para manejo seguro de variables de entorno. |

---

## 3. Posibles Preguntas y Respuestas (FAQ Técnica)

### **Q1: ¿Por qué elegiste LangGraph sobre el AgentExecutor estándar?**
**A:** LangGraph permite definir agentes como grafos de estado. Esto ofrece mayor control sobre los ciclos del agente, permite implementar memorias persistentes de forma más robusta y facilita la gestión de comportamientos complejos que no son lineales.

### **Q2: ¿Cómo aseguras que el agente no responda sobre temas ajenos a viajes?**
**A:** Implementé una doble capa de validación. Primero, una validación por código en `agent.py` que filtra palabras clave (Keyword Filtering). Segundo, un System Prompt restrictivo que instruye al modelo a declinar cortésmente cualquier tema fuera del contexto turístico.

### **Q3: Explica el proceso de RAG implementado.**
**A:** Es un pipeline de conocimiento bajo demanda:
1.  **Ingesta:** Se raspa una fuente web (Wikipedia) usando `BeautifulSoup`.
2.  **Fragmentación:** El texto se divide con `RecursiveCharacterTextSplitter`.
3.  **Indexación:** Se generan embeddings y se almacenan en una base de datos vectorial local (**FAISS**).
4.  **Recuperación:** La tool `travel_knowledge` consulta esta base para enriquecer la respuesta del LLM con datos específicos.

### **Q4: ¿Cómo manejas la latencia y los costos de tokens?**
**A:** Mediante la función `_trim_to_window` que aplica una "ventana deslizante" sobre el historial de mensajes. Esto evita que el prompt crezca indefinidamente, manteniendo los costos bajos y las respuestas rápidas sin perder el contexto reciente.

---

## 4. Diferenciadores de Calidad (Puntos Extra)
*   **Diseño Premium:** Uso de variables CSS, modo oscuro/claro dinámico y efectos de glassmorphism en la interfaz.
*   **Código Limpio:** Separación clara entre la lógica del agente, la API y la interfaz de usuario.
*   **Estado de Producción:** El proyecto está listo para ser contenedorizado y desplegado en plataformas como Docker o Render.
