"""
ViajeBot — Travel Agent (LangGraph ReAct + LangChain 1.x compatible)
Tools: web_search, currency_converter, travel_knowledge (RAG), place_image_search
Memory: MemorySaver with thread_id per session, last 8 non-system messages sent to the model
"""

import os
import re
import requests
from functools import lru_cache
from urllib.parse import unquote
from dotenv import load_dotenv
from langdetect import detect_langs, LangDetectException

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# LangGraph (modern agent API — replaces AgentExecutor in LangChain 1.x)
from langgraph.prebuilt import create_react_agent
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAG_URL = os.getenv("RAG_URL", "https://en.wikipedia.org/wiki/Tourism_in_Colombia")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
GROQ_THINKING_MODEL = os.getenv("GROQ_THINKING_MODEL", "llama-3.3-70b-versatile")

TRAVEL_SCOPE_MESSAGE_ES = (
    "Puedo ayudarte con viajes, destinos, mapas, itinerarios, hoteles, vuelos, "
    "visas, presupuestos, seguridad, gastronomia local y lugares para visitar. "
    "Para mantenerme enfocado, reformula tu pregunta dentro de ese contexto viajero."
)

TRAVEL_KEYWORDS = {
    "travel", "trip", "tourism", "tourist", "destination", "destinations", "city",
    "country", "countries", "visit", "visiting", "itinerary", "flight", "flights",
    "hotel", "hotels", "hostel", "airbnb", "visa", "passport", "budget", "currency",
    "exchange", "rate", "usd", "eur", "cop", "gbp", "mxn", "brl", "weather", "route", "map", "maps", "location", "located", "beach",
    "museum", "restaurant", "food", "safety", "transport", "airport", "train",
    "bus", "packing", "season", "vacation", "holidays", "images", "photos",
    "ticket", "tickets", "price", "prices", "cost", "costs", "fare", "fares", "how much",
    "dangerous", "danger", "safe", "unsafe", "risky", "risk", "crime", "criminal", "violence", "violent", "advice", "warning",
    "advisory", "secure", "security", "warning", "precaution", "cautious", "avoid", "threat",
    "viaje", "viajar", "turismo", "turista", "destino", "destinos", "ciudad",
    "pais", "paises", "visitar", "itinerario", "vuelo", "vuelos", "hotel",
    "hoteles", "hostal", "visa", "pasaporte", "presupuesto", "moneda", "cambio",
    "tasa", "dolar", "dolares", "euro", "euros", "pesos", "precio", "precios",
    "costo", "costos", "tarifa", "tarifas", "boleto", "boletos", "tiquete", "tiquetes",
    "pasaje", "pasajes", "cuanto cuesta", "cuánto cuesta",
    "clima", "ruta", "mapa", "mapas", "ubicacion", "ubicado", "queda", "playa",
    "museo", "restaurante", "comida", "seguridad", "transporte", "aeropuerto",
    "tren", "bus", "empacar", "temporada", "vacaciones", "imagenes", "fotos",
    "lugares", "ciudades", "google maps", "donde esta", "donde queda", "donde se ubica", "llegar", "arrive", "how to", "como llegar",
    "peligroso", "peligroso", "seguro", "peligro", "peligros", "delito", "violencia", "advertencia", "consejo", "recomendacion", "evitar", "riesgo",
    "asia", "africa", "europe", "america", "center america", "central america", "south america", "north america", "oceania", "middle east",
    "europa", "sudamerica", "suramerica", "centroamerica", "norteamerica", "oceania", "medio oriente",
}

KNOWN_DESTINATIONS = {
    "colombia", "bogota", "bogotá", "medellin", "medellín", "cartagena",
    "santa marta", "tayrona", "san andres", "san andrés", "providencia",
    "tokio", "tokyo", "japon", "japón", "kyoto", "osaka", "paris", "londres",
    "madrid", "barcelona", "roma", "venecia", "new york", "nueva york",
    "miami", "mexico", "méxico", "cancun", "cancún", "buenos aires", "lima",
    "cusco", "machu picchu", "rio de janeiro", "rio", "panama", "panamá",
    "costa rica", "chile", "argentina", "peru", "perú", "brasil", "españa",
    "italia", "francia", "alemania", "portugal", "tailandia", "dubai",
    "spain", "japan", "united states", "usa", "canada", "morocco", "egypt",
    "turkey", "greece", "iceland", "norway", "sweden", "finland", "india",
    "vietnam", "indonesia", "philippines", "australia", "new zealand",
    "south africa", "kenya", "tanzania", "namibia", "georgia", "armenia",
    "albania", "montenegro", "slovenia", "croatia", "estonia", "latvia",
    "lithuania", "nepal", "bhutan", "uzbekistan", "kazakhstan", "jordania", "jordan", "venice", "venezia", "italy", "italia",
}

IMAGE_SEARCH_ALIASES = {
    "bogota": "Bogotá, Colombia",
    "bogotá": "Bogotá, Colombia",
    "medellin": "Medellín, Colombia",
    "medellín": "Medellín, Colombia",
    "san andres": "San Andrés, Colombia",
    "san andrés": "San Andrés, Colombia",
    "seville": "Seville, Spain",
    "sevilla": "Seville, Spain",
    "cancun": "Cancún, Mexico",
    "cancún": "Cancún, Mexico",
    "new york": "New York City, USA",
    "nueva york": "New York City, USA",
}


def _get_image_search_variants(query: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", query).strip()
    if not cleaned:
        return []
    normalized = cleaned.lower()
    variants: list[str] = []
    alias = IMAGE_SEARCH_ALIASES.get(normalized)
    if alias:
        variants.append(alias)
    if cleaned not in variants:
        variants.append(cleaned)
    lowercase_variant = cleaned.lower()
    if lowercase_variant != cleaned and lowercase_variant not in [v.lower() for v in variants]:
        variants.append(lowercase_variant)
    return variants

IMAGE_REQUEST_TERMS = ("imagen", "imagenes", "foto", "fotos", "galeria", "gallery", "image", "images", "photo", "photos")
MAP_REQUEST_TERMS = ("mapa", "maps", "google maps", "ubicacion", "ubicado", "donde queda", "donde esta", "location", "located", "where is")
GENERIC_SCOPE_REFUSALS = (
    "i'm here to help with travel",
    "i'm here to help you with travel",
    "if you're interested in visiting",
    "puedo ayudarte con viajes",
)

# ---------------------------------------------------------------------------
# RAG Knowledge Base (loaded at startup)
# ---------------------------------------------------------------------------
_rag_retriever = None


def _retry(operation, attempts: int = 2):
    """Run a small retry loop for flaky network/model calls."""
    last_error = None
    for _ in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
    raise last_error


def _ddg_text_search(query: str, max_results: int = 4) -> list[dict]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _ddg_image_search(query: str, max_results: int = 6) -> list[dict]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        return list(ddgs.images(query, max_results=max_results, safesearch="moderate"))


def setup_rag() -> None:
    """Scrape RAG_URL, chunk, embed, store in FAISS. Called once at startup."""
    global _rag_retriever
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from bs4 import BeautifulSoup

        print(f"[RAG] Fetching content from: {RAG_URL}")
        resp = _retry(lambda: requests.get(RAG_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"}))
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n", strip=True)

        splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=80)
        chunks = splitter.create_documents([raw_text], metadatas=[{"source": RAG_URL}])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        _rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print(f"[RAG] ✅ Indexed {len(chunks)} chunks from {RAG_URL}")
    except Exception as exc:
        print(f"[RAG] ⚠️  RAG setup failed (will skip): {exc}")
        _rag_retriever = None


# ---------------------------------------------------------------------------
# Tool 1 — Web Search (DuckDuckGo, free, no API key)
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time travel information: flights, hotels,
    destinations, visa requirements, travel news, events, and weather.
    Use whenever the user needs current or up-to-date travel details.

    Args:
        query: Search query string.
    """
    try:
        results = _retry(lambda: _ddg_text_search(query, max_results=4))
        if not results:
            return "No search results found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"


# ---------------------------------------------------------------------------
# Tool 2 — Currency Converter (open.er-api.com, free, no API key)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _currency_rates(from_currency: str) -> dict:
    url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
    resp = _retry(lambda: requests.get(url, timeout=10))
    return resp.json()


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert monetary amounts between any two world currencies in real time.
    Essential for travelers to understand costs in their local currency
    or the destination currency.

    Args:
        amount: Numeric amount to convert.
        from_currency: ISO 4217 source code (e.g. 'USD', 'EUR', 'COP').
        to_currency: ISO 4217 target code (e.g. 'COP', 'USD', 'GBP').
    """
    try:
        source = from_currency.upper()
        data = _currency_rates(source)
        if data.get("result") != "success":
            return "Could not retrieve exchange rates. Please try again later."
        rates = data.get("rates", {})
        target = to_currency.upper()
        if target not in rates:
            return (
                f"Currency code '{to_currency}' not recognized. "
                "Use standard codes like USD, EUR, COP, GBP, MXN, BRL."
            )
        converted = amount * rates[target]
        rate = rates[target]
        return (
            f"💱 Currency Conversion\n"
            f"  {amount:,.2f} {source} → {converted:,.2f} {target}\n"
            f"  Rate: 1 {source} = {rate:,.4f} {target}\n"
            f"  Source: open.er-api.com"
        )
    except Exception as exc:
        return f"Conversion error: {exc}"


# ---------------------------------------------------------------------------
# Tool 3 — Travel Knowledge Base (RAG)
# ---------------------------------------------------------------------------
@tool
def travel_knowledge(query: str) -> str:
    """
    Retrieve specialized knowledge about Colombia tourism, national parks,
    cultural destinations, cuisine, and travel tips from the knowledge base.
    Use this for Colombia-specific questions before doing a web search.

    Args:
        query: Travel topic or destination to look up.
    """
    if _rag_retriever is None:
        return "Knowledge base unavailable. Use web_search instead."
    try:
        return _cached_travel_knowledge(query)
    except Exception as exc:
        return f"Retrieval error: {exc}"


@lru_cache(maxsize=128)
def _cached_travel_knowledge(query: str) -> str:
    docs = _retry(lambda: _rag_retriever.invoke(query))
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(d.page_content[:900] for d in docs)


@lru_cache(maxsize=256)
def _cached_destination_images(query: str, max_results: int = 6) -> tuple[str, ...]:
    """
    Search destination images using DuckDuckGo Images.
    Returns direct thumbnail/image URLs suitable for frontend galleries.
    """
    cleaned = re.sub(r"\s+", " ", query).strip()
    if not cleaned:
        return []

    variants = _get_image_search_variants(cleaned)
    try:
        for variant in variants:
            search_query = f"{variant} tourism city landmark"
            results = _retry(lambda: _ddg_image_search(search_query, max_results=max_results))
            images: list[str] = []
            for result in results:
                url = result.get("image") or result.get("thumbnail")
                if (
                    url
                    and url.startswith(("http://", "https://"))
                    and not _is_non_travel_image_url(url)
                    and url not in images
                ):
                    images.append(url)
            if images:
                return tuple(images[:max_results])
    except Exception:
        pass

    for variant in variants:
        images = _wikimedia_destination_images(variant, max_results=max_results)
        if images:
            return tuple(images)

    return tuple()


def search_destination_images(query: str, max_results: int = 6) -> list[str]:
    return list(_cached_destination_images(query, max_results))


@lru_cache(maxsize=256)
def _wikimedia_destination_images(query: str, max_results: int = 6) -> list[str]:
    """Fallback image lookup through Wikipedia/Wikimedia Commons."""
    queries = []
    normalized = query.lower().strip()
    alias = IMAGE_SEARCH_ALIASES.get(normalized)
    if alias:
        queries.append(alias)
    if query not in queries:
        queries.append(query)

    try:
        session = requests.Session()
        headers = {"User-Agent": "ViajeBot/1.0 travel assistant"}

        for search_query in queries:
            search_resp = _retry(lambda: session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": search_query,
                    "format": "json",
                    "srlimit": 1,
                },
                headers=headers,
                timeout=10,
            ))
            search_resp.raise_for_status()
            search_items = search_resp.json().get("query", {}).get("search", [])
            title = search_items[0]["title"] if search_items else search_query

            page_resp = _retry(lambda: session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "pageimages|images",
                    "pithumbsize": 900,
                    "imlimit": 25,
                    "format": "json",
                },
                headers=headers,
                timeout=10,
            ))
            page_resp.raise_for_status()
            pages = page_resp.json().get("query", {}).get("pages", {})

            images: list[str] = []
            file_titles: list[str] = []
            for page in pages.values():
                thumbnail = page.get("thumbnail", {}).get("source")
                if thumbnail and not _is_non_travel_image_url(thumbnail):
                    images.append(thumbnail)
                for item in page.get("images", []):
                    file_title = item.get("title", "")
                    lower = file_title.lower()
                    if any(skip in lower for skip in ("flag", "coat of arms", "map", "icon", ".svg", ".ogg", ".pdf")):
                        continue
                    file_titles.append(file_title)

            for file_title in file_titles[:12]:
                info_resp = _retry(lambda: session.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "titles": file_title,
                        "prop": "imageinfo",
                        "iiprop": "url",
                        "iiurlwidth": 900,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=10,
                ))
                info_resp.raise_for_status()
                info_pages = info_resp.json().get("query", {}).get("pages", {})
                for info_page in info_pages.values():
                    for info in info_page.get("imageinfo", []):
                        url = info.get("thumburl") or info.get("url")
                        if url and url.startswith(("http://", "https://")) and not _is_non_travel_image_url(url):
                            url = unquote(url)
                            if url not in images:
                                images.append(url)
                        if len(images) >= max_results:
                            return images[:max_results]

            if images:
                return images[:max_results]

        return []
    except Exception:
        return []


@lru_cache(maxsize=256)
def fetch_destination_summary(query: str) -> str:
    """Fetch a short encyclopedic destination summary for location fallback responses."""
    cleaned_query = re.sub(r"\s+", " ", query).strip()
    if not cleaned_query:
        return ""
    wiki_title = IMAGE_SEARCH_ALIASES.get(cleaned_query.lower(), cleaned_query)
    wiki_title = wiki_title.replace(" ", "_")
    try:
        resp = _retry(lambda: requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}",
            headers={"User-Agent": "ViajeBot/1.0 travel assistant"},
            timeout=10,
        ))
        if resp.status_code != 200:
            return ""
        data = resp.json()
        extract = data.get("extract", "")
        if not extract:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sentences[:2]).strip()
    except Exception:
        return ""


def _is_non_travel_image_url(url: str) -> bool:
    lower = unquote(url).lower()
    blocked = (
        "flag_of",
        "coat_of_arms",
        "emblem",
        "location_map",
        "relief_location_map",
        "topo",
        "blank_map",
        "wikimedia-logo",
        ".svg",
        ".png/960px-flag",
    )
    return any(term in lower for term in blocked)


@tool
def place_image_search(query: str) -> str:
    """
    Search for images of travel destinations, landmarks, countries, cities,
    beaches, parks, and lesser-known places when the user asks for photos,
    images, gallery, or visual references.

    Args:
        query: Destination, country, city, landmark, or place name.
    """
    images = search_destination_images(query, max_results=5)
    if not images:
        return "No image results found for that destination."
    return f"[INTERNAL DATA] I found {len(images)} images for '{query}'. (Note: Inform the user and mention the gallery below, but DO NOT print these URLs manually): " + ", ".join(images)


# ---------------------------------------------------------------------------
# System Prompt (extended)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are ViajeBot, a warm travel assistant for Colombia and worldwide destinations.

Scope: answer only travel topics: destinations, itineraries, maps, photos, flights, hotels, visas, budgets, currency, food, transport, seasons, safety and travel risks.

Style:
- Match the user's language: Spanish or English.
- Keep responses concise unless user requests details.
- For destination planning, use short sections when useful: Overview, Budget, Safety, Best season, Food, Transportation.

Tools:
- Use travel_knowledge first for Colombia-specific context.
- Use web_search for time-sensitive or uncertain facts: prices, routes, visas, safety alerts, schedules.
- Use currency_converter for exchange requests.
- Use place_image_search when users ask for images/photos/gallery.

Accuracy:
- Never invent exact prices, hotels, routes, visa rules, opening hours or safety facts.
- For safety questions, answer directly with balanced risks and precautions.
- Mention that travel details can change when using current facts.

Memory:
- Remember user preferences in this session: budget, travel style, companions and trip duration.
- Confirm before replacing an existing preference.

Visual UI:
- The frontend renders maps and image galleries. Mention that the map/gallery appears below.
- Do not print raw image URLs.
"""


# ---------------------------------------------------------------------------
# Agent — LangGraph ReAct (modern replacement for AgentExecutor)
# ---------------------------------------------------------------------------
_tools = [web_search, currency_converter, travel_knowledge, place_image_search]
_llm = ChatGroq(model=GROQ_FAST_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
_thinking_llm = ChatGroq(model=GROQ_THINKING_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
_memory = MemorySaver()


def _trim_to_window(messages: list, window: int = 8) -> list:
    """Keep only the last `window` non-system messages for model context."""
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    return other_msgs[-window:] if len(other_msgs) > window else other_msgs


def _agent_prompt(state: dict) -> list:
    """Reduce model context while keeping the latest conversation turns."""
    messages = state.get("messages", [])
    return [SystemMessage(content=SYSTEM_PROMPT)] + _trim_to_window(messages, window=8)


# Build both agents once. They share tools, FAISS-backed RAG and session memory.
_agent = create_react_agent(
    _llm,
    _tools,
    checkpointer=_memory,
    prompt=_agent_prompt,
)
_thinking_agent = create_react_agent(
    _thinking_llm,
    _tools,
    checkpointer=_memory,
    prompt=_agent_prompt,
)


def _select_agent(user_message: str, mode: str = "text"):
    """
    Route simple/voice turns to the fast model and complex text turns to the
    stronger reasoning model. Tool functions and FAISS stay shared.
    """
    normalized = user_message.lower()
    normalized = re.sub(r"\s+", " ", normalized)

    if mode in {"voice", "audio"}:
        return _agent, GROQ_FAST_MODEL

    fast_only_terms = (
        "imagen", "imagenes", "imágenes", "foto", "fotos", "mapa", "maps",
        "google maps", "donde queda", "dónde queda", "donde esta", "dónde está",
        "convierte", "convertir", "currency", "cambio", "tasa",
    )
    if _contains_any(normalized, fast_only_terms):
        return _agent, GROQ_FAST_MODEL

    thinking_terms = (
        "itinerario", "plan", "ruta", "presupuesto", "budget", "seguridad",
        "safety", "riesgo", "visa", "comparar", "compare", "recomienda",
        "recommend", "familia", "family", "dias", "días", "weeks", "semanas",
        "barato", "lujo", "hotel", "hoteles",
    )
    is_long = len(normalized.split()) >= 18
    if is_long or _contains_any(normalized, thinking_terms):
        return _thinking_agent, GROQ_THINKING_MODEL

    return _agent, GROQ_FAST_MODEL


def _is_travel_related(user_message: str) -> bool:
    """Simple hard guardrail before the LLM so off-topic chats do not slip through."""
    normalized = user_message.lower()
    normalized = re.sub(r"[^\w\sáéíóúüñ]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if any(keyword in normalized for keyword in TRAVEL_KEYWORDS):
        return True
    if any(destination in normalized for destination in KNOWN_DESTINATIONS):
        return True
    return False


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return any(term in normalized for term in terms)


def _detect_language(text: str) -> str:
    """
    Detect if text is primarily in Spanish or English.
    Returns: 'es' for Spanish, 'en' for English, or 'es' by default if uncertain.
    """
    if not text or len(text.strip()) < 3:
        return 'es'  # Default to Spanish for very short text
    
    try:
        # langdetect returns a list of Language objects with probabilities
        detections = detect_langs(text)
        if detections:
            # Get the language with highest probability
            primary_lang = str(detections[0]).split(':')[0]  # e.g., 'en' or 'es'
            # Check if it's English, if not assume Spanish
            if primary_lang == 'en':
                return 'en'
            elif primary_lang == 'es':
                return 'es'
    except (LangDetectException, Exception):
        # Fallback to checking for Spanish keywords if detection fails
        normalized = text.lower()
        spanish_markers = (
            "donde", "dónde", "esta", "está", "ubicado", "ubicada", "imagenes",
            "imágenes", "fotos", "muestrame", "muéstrame", "lugares", "viaje",
            "viajar", "pais", "país", "ciudad",
        )
        if any(marker in normalized for marker in spanish_markers):
            return 'es'
    
    return 'es'  # Default to Spanish


def _extract_destination_from_location_question(text: str) -> str | None:
    normalized = text.strip()
    patterns = [
        r"(?:where\s+is|where's)\s+(.+?)(?:\s+located|\s+situated|\?|$)",
        r"(?:location\s+of|map\s+of|google\s+maps\s+of|photos\s+of|images\s+of|pictures\s+of)\s+(?:of\s+|in\s+)?(.+?)(?:\?|$)",
        r"(?:donde\s+esta|dónde\s+está|donde\s+queda|dónde\s+queda|ubicacion\s+de|ubicación\s+de|donde\s+se\s+ubica|dónde\s+se\s+ubica|visitar|viaje\s+a)\s+(?:en\s+|de\s+|a\s+)?(.+?)(?:\?|$)",
        r"(?:donde\s+esta\s+ubicado|dónde\s+está\s+ubicado|donde\s+esta\s+ubicada|dónde\s+está\s+ubicada)\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            destination = re.sub(r"\s+", " ", match.group(1)).strip(" .¿?¡!")
            return destination or None
    return None


def run_agent(session_id: str, user_message: str, mode: str = "text") -> dict:
    """
    Run the agent for a given session.
    Returns: { text, tool_used, tool_name, tools_used }
    """
    # Detect user language early
    user_language = _detect_language(user_message)
    user_is_spanish = (user_language == 'es')
    
    if not _is_travel_related(user_message):
        return {
            "text": TRAVEL_SCOPE_MESSAGE_ES,
            "tool_used": False,
            "tools_used": [],
            "tool_name": None,
        }

    config  = {"configurable": {"thread_id": session_id}}
    agent_message = user_message
    destination_for_location = _extract_destination_from_location_question(user_message)
    
    # Add language hint to the agent
    language_hint = "[RESPOND IN SPANISH]" if user_is_spanish else "[RESPOND IN ENGLISH]"
    
    mode_hint = ""
    if mode in {"voice", "audio"}:
        mode_hint = " [VOICE MODE: answer in 2-4 short sentences.]"

    if destination_for_location:
        agent_message = (
            f"{language_hint}{mode_hint} Travel destination location question. Answer briefly with where this "
            f"place is located and mention the map/gallery below: {user_message}"
        )
    else:
        agent_message = f"{language_hint}{mode_hint} {user_message}"

    inputs  = {"messages": [HumanMessage(content=agent_message)]}

    selected_agent, model_used = _select_agent(user_message, mode=mode)
    result  = _retry(lambda: selected_agent.invoke(inputs, config=config))
    messages: list = result.get("messages", [])

    # ── Extract last AI response ────────────────────────────────────────────
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    output_text = ai_messages[-1].content if ai_messages else "Sorry, I didn't get a response."
    if isinstance(output_text, list):                # handle multi-part content
        output_text = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in output_text
        )

    normalized_output = output_text.lower()
    # user_is_spanish already detected at the beginning of run_agent
    if _contains_any(user_message, IMAGE_REQUEST_TERMS) and (
        "no puedo mostrar" in normalized_output
        or "cannot show" in normalized_output
        or "can't show" in normalized_output
        or "can’t show" in normalized_output
        or "couldn't find specific images" in normalized_output
        or "could not find specific images" in normalized_output
        or "no image results found" in normalized_output
    ):
        if user_is_spanish:
            output_text = (
                "Claro, te muestro una galeria de imagenes abajo en la interfaz. "
                "Tambien puedo ayudarte a elegir barrios, miradores, templos, museos "
                "y zonas seguras para visitar en ese destino."
            )
        else:
            output_text = (
                "Sure, I am showing an image gallery below in the interface. "
                "I can also help you choose neighborhoods, viewpoints, temples, museums, "
                "and safe areas to visit in that destination."
            )
    elif _contains_any(user_message, MAP_REQUEST_TERMS) and (
        "no puedo" in normalized_output
        or "cannot" in normalized_output
        or "can't provide information about the location" in normalized_output
        or "can’t provide information about the location" in normalized_output
    ):
        summary = fetch_destination_summary(destination_for_location) if destination_for_location else ""
        if summary:
            if user_is_spanish:
                output_text = (
                    f"{summary}\n\nAbajo te dejo el mapa de referencia y una galeria de imagenes. "
                    "Desde ahi puedes abrir Google Maps, revisar la ubicacion y calcular rutas."
                )
            else:
                output_text = (
                    f"{summary}\n\nBelow, I am showing the reference map and an image gallery. "
                    "From there you can open Google Maps, check the location, and calculate routes."
                )
        else:
            output_text = (
                "Claro, te dejo el mapa de referencia abajo en la interfaz. "
                "Desde ahi puedes abrir Google Maps, revisar la ubicacion y calcular rutas."
            ) if user_is_spanish else (
                "Sure, I am showing the reference map below in the interface. "
                "From there you can open Google Maps, check the location, and calculate routes."
            )
    elif destination_for_location and any(phrase in normalized_output for phrase in GENERIC_SCOPE_REFUSALS):
        summary = fetch_destination_summary(destination_for_location)
        if summary:
            output_text = (
                f"{summary}\n\nAbajo te muestro el mapa y una galeria de imagenes "
                "para que tengas una referencia visual del lugar. Tambien puedo ayudarte "
                "con zonas recomendadas, temporada ideal, seguridad y presupuesto."
            ) if user_is_spanish else (
                f"{summary}\n\nBelow, I am showing the map and an image gallery so you have "
                "a visual reference for the place. I can also help with recommended areas, "
                "best season, safety, and budget."
            )
        else:
            output_text = (
                f"{destination_for_location} es un destino o lugar que puedes ubicar en el mapa "
                "que aparece abajo en la interfaz. Tambien te muestro una galeria de imagenes "
                "para que tengas una referencia visual del lugar. Si quieres, puedo ayudarte "
                "con mejores zonas para visitar, temporada ideal, seguridad y presupuesto."
            ) if user_is_spanish else (
                f"{destination_for_location} is a destination or place you can locate on the map "
                "shown below in the interface. I am also showing an image gallery for visual context. "
                "I can help with the best areas to visit, ideal season, safety, and budget."
            )

    if _contains_any(user_message, IMAGE_REQUEST_TERMS) and (
        "![" in output_text or "upload.wikimedia.org" in output_text or "http://" in output_text or "https://" in output_text
    ):
        output_text = (
            "Claro, abajo te muestro una galeria visual del destino con imagenes encontradas. "
            "Tambien puedo ayudarte con los lugares mas fotogenicos, mejores zonas para hospedarte "
            "y una ruta para visitarlo."
        ) if user_is_spanish else (
            "Sure, below I am showing a visual gallery of the destination with images I found. "
            "I can also help with the most photogenic places, best areas to stay, and a route to visit it."
        )

    # ── Detect tools used in THIS turn (after last HumanMessage) ───────────
    last_human_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human_idx = i

    tools_used: list[str] = []
    if last_human_idx >= 0:
        for m in messages[last_human_idx:]:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name:
                        tools_used.append(name)

    # ── Final Fallback for destination (from tool calls) ──────────────────
    if not destination_for_location and tools_used:
        for m in messages[last_human_idx:]:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    t_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if t_name == "place_image_search":
                        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                        destination_for_location = args.get("query")
                        break

    return {
        "text":       output_text,
        "tool_used":  len(tools_used) > 0,
        "tools_used": tools_used,
        "tool_name":  tools_used[0] if tools_used else None,
        "destination": destination_for_location,
        "model_used": model_used,
    }
