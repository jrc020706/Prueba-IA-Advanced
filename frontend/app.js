/**
 * ViajeBot — Frontend Logic
 * Handles: chat API, TTS audio, mode toggle, tool badge rendering, chat history.
 */

const API_BASE = 'https://viagebot-backend.onrender.com';

// ── State ──────────────────────────────────────────────────────────────────
let responseMode = 'text';          // 'text' | 'voice'
let sessionId    = generateSessionId();
let isLoading    = false;
let mediaRecorder = null;
let audioChunks   = [];
let isListening   = false;
let activeAudio   = null;
let lastSpeechError = false;

// ── DOM References ─────────────────────────────────────────────────────────
const chatArea    = document.getElementById('chat-area');
const inputEl     = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const micBtn      = document.getElementById('mic-btn');
const btnText     = document.getElementById('btn-text');
const btnVoice    = document.getElementById('btn-voice');
const welcomeCard = document.getElementById('welcome-card');
const statusDot   = document.getElementById('status-dot');
const voiceStatus = document.getElementById('voice-status');
const themeToggle = document.getElementById('theme-toggle');

const DESTINATION_IMAGES = {
  tokio: ['https://images.unsplash.com/photo-1542051841857-5f90071e7989?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=700&q=80'],
  tokyo: ['https://images.unsplash.com/photo-1542051841857-5f90071e7989?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=700&q=80'],
  kyoto: ['https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1528360983277-13d401cdc186?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1545569341-9eb8b30979d9?auto=format&fit=crop&w=700&q=80'],
  cartagena: ['https://images.unsplash.com/photo-1583531352515-8884af319dc1?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1624806992066-5ffcf7ca186b?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1600195077909-46e573870d99?auto=format&fit=crop&w=700&q=80'],
  medellin: ['https://images.unsplash.com/photo-1581781870027-04212e231e96?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1607990281513-2c110a25bd8c?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1534269222346-5a896154c41d?auto=format&fit=crop&w=700&q=80'],
  bogota: ['https://images.unsplash.com/photo-1568632234157-ce7aecd03d0d?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1596120236172-231999844ade?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1599708153386-62bf3f03555f?auto=format&fit=crop&w=700&q=80'],
  paris: ['https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?auto=format&fit=crop&w=700&q=80'],
  "costa rica": ['https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1585208798174-6cedd86e019a?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1598881034666-a85c6f5c3e22?auto=format&fit=crop&w=700&q=80'],
  roma: ['https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1529260830199-42c24126f198?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1529154036614-a60975f5c760?auto=format&fit=crop&w=700&q=80'],
  rome: ['https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1529260830199-42c24126f198?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1529154036614-a60975f5c760?auto=format&fit=crop&w=700&q=80'],
  venecia: ['https://images.unsplash.com/photo-1514890547357-a9ee288728e0?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1534113414509-0eec2bfb493f?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?auto=format&fit=crop&w=700&q=80'],
  venice: ['https://images.unsplash.com/photo-1514890547357-a9ee288728e0?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1534113414509-0eec2bfb493f?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?auto=format&fit=crop&w=700&q=80'],
  espana: ['https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1543783207-ec64e4d95325?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=700&q=80'],
  spain: ['https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1543783207-ec64e4d95325?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=700&q=80'],
  barcelona: ['https://images.unsplash.com/photo-1583422409516-2895a77efded?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1579282240050-352db0a14c61?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1562883676-8c7feb83f09b?auto=format&fit=crop&w=700&q=80'],
  jordania: ['https://images.unsplash.com/photo-1547234935-80c7145ec969?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1501233339699-2051864573af?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1526481280693-3bfa7568e0f3?auto=format&fit=crop&w=700&q=80'],
  jordan: ['https://images.unsplash.com/photo-1547234935-80c7145ec969?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1501233339699-2051864573af?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1526481280693-3bfa7568e0f3?auto=format&fit=crop&w=700&q=80'],
  dubai: ['https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1582672751936-b20e4f9c5e02?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1573843981267-be1999ff37cd?auto=format&fit=crop&w=700&q=80'],
  bali: ['https://images.unsplash.com/photo-1537996194471-e657df975ab4?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1552733407-5d5c46c3bb3b?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1518548419970-58e3b4079ab2?auto=format&fit=crop&w=700&q=80'],
  santorini: ['https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1601581875309-fafbf2d3ed3a?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=700&q=80'],
  grecia: ['https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1601581875309-fafbf2d3ed3a?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=700&q=80'],
  greece: ['https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1601581875309-fafbf2d3ed3a?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=700&q=80'],
  "machu picchu": ['https://images.unsplash.com/photo-1526392060635-9d6019884377?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1580502304784-8985b7eb7260?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1461863109726-246fa9598dc3?auto=format&fit=crop&w=700&q=80'],
  peru: ['https://images.unsplash.com/photo-1526392060635-9d6019884377?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1580502304784-8985b7eb7260?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1461863109726-246fa9598dc3?auto=format&fit=crop&w=700&q=80'],
  "new york": ['https://images.unsplash.com/photo-1500916434205-0c77489c6cf7?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1522083165195-3424ed129620?auto=format&fit=crop&w=700&q=80'],
  "cape town": ['https://images.unsplash.com/photo-1580060839134-75a5edca2e99?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1563656157432-67560011e209?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1504893524553-b855bce32c67?auto=format&fit=crop&w=700&q=80'],
  amsterdam: ['https://images.unsplash.com/photo-1534351590666-13e3e96b5017?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1589308078059-be1415eab4c3?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1583416750470-965b2707b355?auto=format&fit=crop&w=700&q=80'],
  bangkok: ['https://images.unsplash.com/photo-1508009603885-50cf7c579365?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1563492065599-3520f775eeed?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1570366583862-f91883984fde?auto=format&fit=crop&w=700&q=80'],
  marrakech: ['https://images.unsplash.com/photo-1489493887464-892be6d1daae?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?auto=format&fit=crop&w=700&q=80', 'https://images.unsplash.com/photo-1548453457-b5e49c3fb0a2?auto=format&fit=crop&w=700&q=80']
}

const COMMON_DESTINATIONS = [
  'spain', 'espana', 'españa', 'japan', 'japon', 'colombia', 'france', 'francia',
  'italy', 'italia', 'germany', 'alemania', 'portugal', 'morocco', 'marruecos',
  'egypt', 'egipto', 'turkey', 'turquia', 'turquía', 'greece', 'grecia',
  'iceland', 'islandia', 'norway', 'noruega', 'sweden', 'suecia', 'finland',
  'finlandia', 'india', 'vietnam', 'indonesia', 'philippines', 'filipinas',
  'australia', 'new zealand', 'nueva zelanda', 'south africa', 'sudafrica',
  'sudáfrica', 'kenya', 'tanzania', 'namibia', 'georgia', 'armenia', 'albania',
  'montenegro', 'slovenia', 'eslovenia', 'croatia', 'croacia', 'estonia',
  'latvia', 'letonia', 'lithuania', 'lituania', 'nepal', 'bhutan', 'butan',
  'bután', 'uzbekistan', 'uzbekistán', 'kazakhstan', 'kazajistan', 'kazajistán',
  'tokyo', 'tokio', 'kyoto', 'osaka', 'paris', 'london', 'londres', 'madrid',
  'barcelona', 'rome', 'roma', 'venice', 'venecia', 'cartagena', 'bogota',
  'bogotá', 'medellin', 'medellín', 'costa rica', 'dubai', 'jordania', 'jordan',
  'bali', 'santorini', 'marrakech', 'bangkok', 'amsterdam', 'cape town',
  'machu picchu', 'new york', 'nueva york', 'rio de janeiro', 'rio',
  'buenos aires', 'lima', 'cusco', 'cancun', 'cancún', 'mexico', 'méxico',
  'miami', 'panama', 'panamá', 'peru', 'perú', 'chile', 'argentina', 'brasil',
  'moscu', 'moscú', 'moscow', 'rotterdam',
];

// ── Helpers ────────────────────────────────────────────────────────────────
function generateSessionId() {
  return 'sess_' + Math.random().toString(36).slice(2, 10);
}

function setLoading(state) {
  isLoading = state;
  sendBtn.disabled = state;
  inputEl.disabled = state;
  micBtn.disabled = state;
}

function hideWelcome() {
  if (welcomeCard && welcomeCard.parentElement) {
    welcomeCard.style.opacity = '0';
    welcomeCard.style.transform = 'translateY(-8px)';
    welcomeCard.style.transition = 'all 0.3s ease';
    setTimeout(() => welcomeCard.remove(), 300);
  }
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

/** Maps tool name to a human-readable label + emoji */
function toolLabel(toolName) {
  const map = {
    web_search:          '🔍 Web Search',
    currency_converter:  '💱 Currency Converter',
    travel_knowledge:    '📚 Travel Knowledge',
  };
  return map[toolName] || `🔧 ${toolName}`;
}

function normalizeText(text) {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
}

function extractDestination(text) {
  const normalized = normalizeText(text);
  const known = Object.keys(DESTINATION_IMAGES).find(place => normalized.includes(place));
  if (known) return known;
  const common = COMMON_DESTINATIONS
    .sort((a, b) => b.length - a.length)
    .find(place => normalized.includes(normalizeText(place)));
  if (common) return common;

  const patterns = [
    /(?:where\s+is|where's)\s+([a-z\s]{2,50}?)(?:\s+located|\s+situated|\s+on\s+the\s+map|\?|$)/i,
    /(?:location\s+of|map\s+of|google\s+maps\s+of|photos\s+of|images\s+of|pictures\s+of)\s+([a-z\s]{2,50})/i,
    /(?:donde esta ubicado|donde esta ubicada|donde queda ubicado|donde queda ubicada|ubicacion de|ubicacion en)\s+(?:de\s+|en\s+|a\s+)?([a-z\s]{2,50})/i,
    /(?:imagenes|fotos|mapa|maps|ubicado|ubicacion|donde queda|donde esta|visitar|viaje a|viajar a|lugares en|lugares de)\s+(?:de\s+|en\s+|a\s+)?([a-z\s]{3,40})/i,
    /(?:show|images|photos|map|location|visit|travel to|places in)\s+(?:of\s+|in\s+|to\s+)?([a-z\s]{3,40})/i,
  ];

  for (const pattern of patterns) {
    const match = normalizeText(text).match(pattern);
    if (match?.[1]) {
      return cleanDestinationName(match[1]);
    }
  }
  return null;
}

function cleanDestinationName(value) {
  return normalizeText(value)
    .replace(/\b(por favor|please|google maps|maps|mapa|imagenes|imagen|fotos|foto|galeria|muestrame|mostrar|located|situated|ubicado|ubicada|ubica|ubicacion|queda|location|on the map|travel|destination|donde queda|donde esta|visitar|lugares)\b/g, '')
    .replace(/^(de|del|la|el|en|a)\s+/g, '')
    .replace(/\s+(de|del|la|el|en|a)$/g, '')
    .replace(/[¿?¡!.,]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function buildVisualContext(userText, backendDestination = null) {
  const normalized = normalizeText(userText);
  const destination = backendDestination ? cleanDestinationName(backendDestination) : extractDestination(userText);
  if (!destination) return null;

  // If the backend explicitly gave us a destination, we likely want to show SOMETHING.
  const wantsMap = /(mapa|maps|google maps|ubicacion|ubicado|donde queda|donde esta|location|located|where is)/i.test(normalized);
  return {
    destination,
    map: wantsMap,
    images: true,
  };
}

function imageSetFor(destination) {
  const normalized = normalizeText(destination);
  if (DESTINATION_IMAGES[normalized]) return DESTINATION_IMAGES[normalized];
  return [];
}

async function fetchDestinationImages(destination) {
  try {
    const res = await fetch(`${API_BASE}/destination-media`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ destination }),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.images) ? data.images : [];
  } catch {
    return [];
  }
}

async function hydrateImageGallery(row, destination) {
  const gallery = row.querySelector('.image-gallery');
  if (!gallery) return;
  const hasLocalImages = Boolean(gallery.querySelector('img'));
  const images = await fetchDestinationImages(destination);
  if (!images.length) {
    if (hasLocalImages) return;
    gallery.classList.add('image-gallery-empty');
    gallery.innerHTML = `<div>No encontre imagenes especificas para ${escapeHtml(destination)}.</div>`;
    return;
  }

  gallery.classList.remove('image-gallery-loading', 'image-gallery-empty');
  gallery.innerHTML = images.slice(0, 6).map((src, index) => `
    <img src="${src}" alt="${escapeHtml(destination)} travel view ${index + 1}" loading="lazy" />
  `).join('');
}

// ── Mode Toggle ────────────────────────────────────────────────────────────
btnText.addEventListener('click', () => setMode('text'));
btnVoice.addEventListener('click', () => {
  setMode('voice');
  toggleSpeechRecognition();
});
themeToggle.addEventListener('click', toggleTheme);

function setMode(mode) {
  if (mode === 'text' && isListening) {
    recognition?.stop();
  }
  responseMode = mode;
  btnText.classList.toggle('active', mode === 'text');
  btnVoice.classList.toggle('active', mode === 'voice');
  btnText.setAttribute('aria-pressed', mode === 'text');
  btnVoice.setAttribute('aria-pressed', mode === 'voice');
  micBtn.classList.toggle('voice-mode', mode === 'voice');
}

function toggleTheme() {
  const light = document.body.classList.toggle('light-theme');
  themeToggle.setAttribute('aria-label', light ? 'Switch to dark mode' : 'Switch to light mode');
  themeToggle.querySelector('span').textContent = light ? '🌙' : '☀️';
}

function setVoiceStatus(message, tone = 'muted') {
  voiceStatus.textContent = message;
  voiceStatus.className = `voice-status ${tone}`;
}

// ── Quick Chips ────────────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const text = chip.textContent.replace(/^[\u{1F300}-\u{1FFFF}]\s*/u, '').trim();
    inputEl.value = text;
    sendMessage();
  });
});

// ── Input Auto-resize ──────────────────────────────────────────────────────
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 130) + 'px';
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
micBtn.addEventListener('click', toggleSpeechRecognition);

document.querySelectorAll('.destination-card').forEach(card => {
  card.addEventListener('click', () => {
    inputEl.value = card.dataset.prompt;
    sendMessage();
  });
});

function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return null;

  const speech = new SpeechRecognition();
  speech.lang = 'es-CO';
  speech.interimResults = true;
  speech.continuous = false;

  speech.addEventListener('result', event => {
    lastSpeechError = false;
    const transcript = Array.from(event.results)
      .map(result => result[0].transcript)
      .join('');
    inputEl.value = transcript;
    inputEl.dispatchEvent(new Event('input'));
    setVoiceStatus('Escuchando... puedes enviar cuando termine la frase.', 'active');
  });

  speech.addEventListener('end', () => {
    isListening = false;
    micBtn.classList.remove('listening');
    micBtn.title = 'Dictate';
    if (lastSpeechError) return;
    if (inputEl.value.trim()) {
      setVoiceStatus('Texto reconocido. Presiona enviar o Enter.', 'ok');
    } else if (responseMode === 'voice') {
      setVoiceStatus('Pulsa el microfono para dictar tu pregunta de viaje.', 'muted');
    } else {
      setVoiceStatus('');
    }
  });

  speech.addEventListener('error', event => {
    lastSpeechError = true;
    isListening = false;
    micBtn.classList.remove('listening');
    const messages = {
      'not-allowed': 'Permiso de microfono bloqueado. Activalo en el navegador y vuelve a intentar.',
      'audio-capture': 'No encontre un microfono disponible en este dispositivo.',
      'no-speech': 'No escuche voz. Pulsa el microfono e intenta hablar un poco mas cerca.',
      network: 'El reconocimiento de voz necesita conexion del navegador. Intenta de nuevo.',
    };
    setVoiceStatus(messages[event.error] || 'No pude iniciar el reconocimiento de voz en este navegador.', 'error');
  });

  return speech;
}

async function toggleSpeechRecognition() {
  setMode('voice');
  
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setVoiceStatus('Tu navegador no soporta grabacion de audio o estas en una conexion no segura (HTTP).', 'error');
    return;
  }

  if (isListening) {
    stopRecording();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startRecording(stream);
  } catch (error) {
    isListening = false;
    micBtn.classList.remove('listening');
    let msg = 'No pude activar el microfono. Revisa permisos.';
    if (error?.name === 'NotAllowedError') msg = 'Permiso de microfono denegado. Activalo en el navegador.';
    if (error?.name === 'NotFoundError') msg = 'No se encontro ningun microfono conectado.';
    setVoiceStatus(msg, 'error');
  }
}

function startRecording(stream) {
  isListening = true;
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  
  mediaRecorder.ondataavailable = event => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    setVoiceStatus('Procesando voz con Groq...', 'active');
    await sendAudioToBackend(audioBlob);
    
    // Stop all tracks to release microphone
    stream.getTracks().forEach(track => track.stop());
  };

  mediaRecorder.start();
  micBtn.classList.add('listening');
  micBtn.title = 'Stop recording';
  setVoiceStatus('Escuchando... pulsa de nuevo para transcribir.', 'active');
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isListening = false;
  micBtn.classList.remove('listening');
  micBtn.title = 'Dictate';
}

async function sendAudioToBackend(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'recording.wav');

  try {
    const res = await fetch(`${API_BASE}/transcribe`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Transcription failed');
    
    const data = await res.json();
    if (data.text) {
      inputEl.value = data.text;
      inputEl.dispatchEvent(new Event('input'));
      setVoiceStatus('Texto reconocido con Groq.', 'ok');
      // Optional: auto-send if you want
      // sendMessage();
    }
  } catch (error) {
    setVoiceStatus('Error al transcribir audio: ' + error.message, 'error');
  }
}

// ── Render User Bubble ─────────────────────────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="avatar user-avatar" aria-hidden="true">👤</div>
    <div class="bubble-wrapper">
      <div class="bubble user">${escapeHtml(text)}</div>
    </div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

// ── Typing Indicator ───────────────────────────────────────────────────────
function showTyping() {
  const row = document.createElement('div');
  row.className = 'message-row';
  row.id = 'typing-row';
  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">✈️</div>
    <div class="bubble-wrapper">
      <div class="typing-indicator" aria-label="ViajeBot is typing">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

function removeTyping() {
  document.getElementById('typing-row')?.remove();
}

// ── Render Bot Bubble ──────────────────────────────────────────────────────
function appendBotMessage(text, toolUsed, toolName, audioBlob, visualContext = null, audioFailed = false) {
  const row = document.createElement('div');
  row.className = 'message-row';

  const bubbleClass = toolUsed ? 'bubble bot tool-active' : 'bubble bot';

  // Tool badge HTML (persists in history)
  const badgeHtml = toolUsed && toolName
    ? `<div class="tool-badge" aria-label="Tool used: ${toolLabel(toolName)}">
         <span class="tool-dot" aria-hidden="true"></span>
         ${toolLabel(toolName)}
       </div>`
    : '';

  // Audio player HTML
  let audioHtml = '';
  let audioUrl = null;
  if (audioBlob) {
    audioUrl = URL.createObjectURL(audioBlob);
    audioHtml = `
      <div class="audio-player-wrap">
        <button class="play-btn" aria-label="Play audio response">▶</button>
        <span class="audio-label">🔊 Voice response</span>
      </div>
    `;
  }
  if (audioFailed) {
    audioHtml = `
      <div class="audio-player-wrap audio-error">
        <span class="audio-label">Voice mode is active, but audio could not be generated. Check the backend /tts endpoint and OPENAI_API_KEY.</span>
      </div>
    `;
  }

  let visualHtml = '';
  if (visualContext?.map) {
    const query = encodeURIComponent(visualContext.destination);
    visualHtml += `
      <div class="map-card">
        <iframe
          title="Google Maps location for ${escapeHtml(visualContext.destination)}"
          src="https://www.google.com/maps?q=${query}&output=embed"
          loading="lazy"
          referrerpolicy="no-referrer-when-downgrade"></iframe>
        <div class="map-actions">
          <a href="https://www.google.com/maps/search/?api=1&query=${query}" target="_blank" rel="noopener">Open map</a>
          <a href="https://www.google.com/maps/dir/?api=1&destination=${query}" target="_blank" rel="noopener">Directions</a>
        </div>
      </div>
    `;
  }

  if (visualContext?.images) {
    const images = imageSetFor(visualContext.destination);
    visualHtml += `
      <div class="image-gallery ${images.length ? '' : 'image-gallery-loading'}" aria-label="Travel images for ${escapeHtml(visualContext.destination)}">
        ${images.length ? images.map((src, index) => `
          <img src="${src}" alt="${escapeHtml(visualContext.destination)} travel view ${index + 1}" loading="lazy" />
        `).join('') : '<div>Buscando imagenes del destino...</div>'}
      </div>
    `;
  }

  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">✈️</div>
    <div class="bubble-wrapper">
      ${badgeHtml}
      <div class="${bubbleClass}">
        ${formatMarkdown(text)}
        ${visualHtml}
        ${audioHtml}
      </div>
    </div>
  `;
  chatArea.appendChild(row);
  if (visualContext?.images) {
    hydrateImageGallery(row, visualContext.destination);
  }
  if (audioUrl) {
    const playBtn = row.querySelector('.play-btn');
    playBtn.addEventListener('click', () => playAudio(audioUrl, playBtn));
    setTimeout(() => playAudio(audioUrl, playBtn), 200);
  }
  scrollToBottom();
}

// ── Audio Play Button Callback ─────────────────────────────────────────────
function playAudio(url, btn) {
  if (activeAudio?.url === url) {
    if (activeAudio.audio.paused) {
      activeAudio.audio.play().catch(() => {});
      btn.textContent = '⏸';
    } else {
      activeAudio.audio.pause();
      btn.textContent = '▶';
    }
    return;
  }

  if (activeAudio) {
    activeAudio.audio.pause();
    activeAudio.audio.currentTime = 0;
    activeAudio.btn.textContent = '▶';
  }

  const audio = new Audio(url);
  activeAudio = { url, audio, btn };
  btn.textContent = '⏸';
  audio.play().catch(() => { btn.textContent = '▶'; });
  audio.addEventListener('ended', () => {
    btn.textContent = '▶';
    if (activeAudio?.url === url) activeAudio = null;
  });
}

// ── Simple Markdown Formatter ──────────────────────────────────────────────
function formatMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px;font-size:0.85em">$1</code>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Main Send Logic ────────────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;

  hideWelcome();
  appendUserMessage(text);

  // Reset input
  inputEl.value = '';
  inputEl.style.height = 'auto';
  setLoading(true);
  showTyping();

  try {
    // 1. Send to /chat
    const chatRes = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message:    text,
        session_id: sessionId,
        mode:       responseMode,
      }),
    });

    if (!chatRes.ok) {
      const err = await chatRes.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${chatRes.status}`);
    }

    const data = await chatRes.json();
    // data: { text, tool_used, tool_name, tools_used, mode }

    // 2. TTS if voice mode
    let audioBlob = null;
    let audioFailed = false;
    if (responseMode === 'voice') {
      try {
        const ttsRes = await fetch(`${API_BASE}/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: data.text }),
        });
        if (ttsRes.ok) {
          audioBlob = await ttsRes.blob();
        } else {
          audioFailed = true;
        }
      } catch {
        // TTS failure is non-fatal — still show text
        audioFailed = true;
      }
    }

    removeTyping();
    const visualDestination = data.destination || text;
    appendBotMessage(data.text, data.tool_used, data.tool_name, audioBlob, buildVisualContext(text, data.destination), audioFailed);

  } catch (err) {
    removeTyping();
    appendBotMessage(
      `⚠️ Sorry, I encountered an error: ${err.message}\nMake sure the backend is running at ${API_BASE}.`,
      false, null, null
    );
    // Red status dot on error
    statusDot.style.background = '#ff6b6b';
    statusDot.style.boxShadow  = '0 0 0 2px rgba(255,107,107,0.3)';
  } finally {
    setLoading(false);
  }
}

// ── Health Check on Load ───────────────────────────────────────────────────
(async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error();
    // Green — already default
  } catch {
    statusDot.style.background = '#ff6b6b';
    statusDot.style.boxShadow  = '0 0 0 2px rgba(255,107,107,0.3)';
    statusDot.title = 'Backend offline';
  }
})();

// ── Floating Chat Popup Logic ──────────────────────────────────────────────
const chatFab = document.getElementById('chat-fab');
const chatPopup = document.getElementById('chat-popup');

if (chatFab && chatPopup) {
  chatFab.addEventListener('click', () => {
    chatFab.classList.toggle('is-open');
    if (chatFab.classList.contains('is-open')) {
      chatFab.querySelector('.fab-open').style.display = 'none';
      chatFab.querySelector('.fab-close').style.display = 'inline';
      chatPopup.classList.add('is-open');
      // Autofocus input when opening
      setTimeout(() => document.getElementById('user-input').focus(), 300);
      
      // Hide badge
      const badge = document.getElementById('fab-badge');
      if(badge) badge.classList.remove('show');
    } else {
      chatFab.querySelector('.fab-open').style.display = 'inline';
      chatFab.querySelector('.fab-close').style.display = 'none';
      chatPopup.classList.remove('is-open');
    }
  });
}

// Ensure clicking destination gallery cards sends the message
document.querySelectorAll('.destination-card').forEach(card => {
  card.addEventListener('click', () => {
    const prompt = card.getAttribute('data-prompt');
    if (!prompt) return;

    inputEl.value = prompt;
    inputEl.dispatchEvent(new Event('input'));

    if (chatFab && !chatFab.classList.contains('is-open')) {
      chatFab.click();
    }

    setTimeout(() => {
      if (inputEl.value.trim()) {
        sendMessage();
        inputEl.value = '';
        inputEl.style.height = 'auto';
      }
    }, 250);
  });
});
