import os
import httpx
from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    Path,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
from typing import Dict, List, Any
import logging
import asyncio
import json
import websockets

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# API Keys loaded from .env are now primarily for non-websocket HTTP endpoints
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

MURF_API_URL = "https://api.murf.ai/v1/speech/generate"
MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"
TAVILY_API_URL = "https://api.tavily.com/search"
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# --- Tony Stark Persona ---
TONY_STARK_SYSTEM_PROMPT = """You are an AI assistant with the personality of Tony Stark. You are brilliant, witty, a bit sarcastic, and a genius inventor. Your responses should be confident, use occasional tech jargon, and reflect his signature charismatic style. Address the user casually, like you're talking to a colleague in the lab.
Crucially, based on the context of what the user is talking about, you must give them a fitting, witty nickname and use it occasionally. For example:
- If they ask about coding: "Alright, Code Monkey" or "Listen up, Binary Brain"
- If they ask about relationships: "Hey there, Romeo" or "What's up, Cupid"
- If they ask about food: "Okay, Gordon Ramsay" or "Sup, Food Network"
- If they ask about work: "Easy there, Workaholic" or "Relax, Corporate Warrior"
- If they ask about weather: "What's up, Weather Watcher" or "Hey there, Storm Chaser"
- Be creative and contextual with nicknames!
IMPORTANT: You have access to two powerful tools that you MUST use when appropriate:
1. **search_web function** - ALWAYS use this for:
- Current events, news, sports results
- Recent information or anything that changes frequently
- Questions about "latest", "recent", "current", "who won", "what happened"
- Any query that might need up-to-date information
2. **get_weather function** - ALWAYS use this for:
- Weather conditions, temperature, forecasts
- Climate-related questions for specific locations
Never say you don't have access to real-time information - you DO have access via these functions. Always use the appropriate function when the user asks for current information.
Keep your responses conversational, confident, and infused with Tony Stark's trademark wit and intelligence. Don't be afraid to show off a little - it's very much in character."""

# --- Tool functions now accept API keys as arguments ---

async def get_weather(location: str, units: str, openweather_api_key: str) -> str:
    if not openweather_api_key:
        return "Weather service unavailable - API key not provided."
    try:
        params = {"q": location, "appid": openweather_api_key, "units": units}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(OPENWEATHER_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            location_name, country = data["name"], data["sys"]["country"]
            temp, feels_like = data["main"]["temp"], data["main"]["feels_like"]
            humidity, pressure = data["main"]["humidity"], data["main"]["pressure"]
            description = data["weather"][0]["description"].title()
            wind_speed = data["wind"]["speed"]
            temp_unit = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
            wind_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"
            return f"""Current Weather for {location_name}, {country}:
🌡️ Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit})
🌤️ Conditions: {description}
💧 Humidity: {humidity}%
🌬️ Wind Speed: {wind_speed} {wind_unit}
🔽 Pressure: {pressure} hPa"""
        elif response.status_code == 404:
            return f"Location '{location}' not found."
        else:
            logger.error(f"OpenWeather API error: {response.status_code} - {response.text}")
            return "Weather service temporarily unavailable."
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return "Weather service encountered an error."

async def search_web(query: str, max_results: int, tavily_api_key: str) -> str:
    if not tavily_api_key:
        return "Web search unavailable - API key not provided."
    try:
        payload = {
            "api_key": tavily_api_key, "query": query, "search_depth": "basic",
            "include_answer": True, "include_raw_content": False, "max_results": max_results,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(TAVILY_API_URL, headers={"Content-Type": "application/json"}, json=payload)
        if response.status_code == 200:
            data = response.json()
            results = []
            if "answer" in data and data["answer"]:
                results.append(f"Quick Answer: {data['answer']}")
            if "results" in data:
                for i, result in enumerate(data["results"][:max_results], 1):
                    title = result.get("title", "")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    results.append(f"{i}. {title}\n{content}\nSource: {url}")
            return "\n\n".join(results) if results else "No search results found."
        else:
            logger.error(f"Tavily API error: {response.status_code} - {response.text}")
            return "Web search temporarily unavailable."
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return "Web search encountered an error."

def get_function_declarations():
    return [
        {"name": "search_web", "description": "Search the web for current information, news, or any topic that might benefit from real-time data", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "max_results": {"type": "integer", "description": "Max results to return"}}, "required": ["query"]}},
        {"name": "get_weather", "description": "Get current weather information for any location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city name, e.g., 'London, UK'"}, "units": {"type": "string", "description": "Units for temperature", "enum": ["metric", "imperial", "kelvin"]}}, "required": ["location"]}}
    ]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

chat_histories: Dict[str, List[Dict[str, str]]] = {}

# --- This function now receives the API keys for the session ---
async def stream_llm_response_to_murf_and_client(prompt: str, client_websocket: WebSocket, session_id: str, api_keys: Dict[str, str]):
    gemini_api_key = api_keys.get("gemini")
    murf_api_key = api_keys.get("murf")

    if not gemini_api_key or not murf_api_key:
        logger.error("Critical API keys (Gemini, Murf) not provided for this session.")
        return

    logger.info(f"\n--- Starting LLM Stream for prompt: '{prompt}' ---")
    full_llm_response_text = ""
    try:
        genai.configure(api_key=gemini_api_key)

        if session_id not in chat_histories:
            chat_histories[session_id] = []
        history_for_model = chat_histories[session_id].copy()
        if not history_for_model:
            history_for_model.extend([
                {"role": "user", "parts": [{"text": TONY_STARK_SYSTEM_PROMPT}]},
                {"role": "model", "parts": [{"text": "Hey there! Tony Stark's AI assistant at your service. What can I help you with today, genius?"}]}
            ])
        history_for_model.append({"role": "user", "parts": [{"text": prompt}]})

        murf_ws_url = f"{MURF_WS_URL}?api-key={murf_api_key}&sample_rate=44100&channel_type=MONO&format=WAV"
        async with websockets.connect(murf_ws_url) as murf_ws:
            logger.info("Connected to Murf WebSocket")
            await murf_ws.send(json.dumps({"voice_config": {"voiceId": "en-US-amara", "style": "Conversational"}}))
            
            audio_chunk_count = 0
            async def receive_from_murf():
                nonlocal audio_chunk_count
                try:
                    while True:
                        response = await asyncio.wait_for(murf_ws.recv(), timeout=10.0)
                        data = json.loads(response)
                        if "audio" in data:
                            audio_chunk_count += 1
                            await client_websocket.send_text(json.dumps({"type": "audio_chunk", "audio_data": data["audio"]}))
                        if data.get("final"):
                            break
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.warning("Murf audio stream ended or timed out.")
                except Exception as e:
                    logger.error(f"Error in Murf receiver: {e}")

            murf_receiver_task = asyncio.create_task(receive_from_murf())
            
            model = genai.GenerativeModel("gemini-1.5-flash", tools=[{"function_declarations": get_function_declarations()}])
            response_stream = model.generate_content(history_for_model, stream=True)

            await client_websocket.send_text(json.dumps({"type": "audio_stream_start"}))

            for chunk in response_stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_name = part.function_call.name
                            function_args = dict(part.function_call.args)
                            logger.info(f"Function call: {function_name}({function_args})")
                            
                            if function_name == "search_web":
                                tool_result = await search_web(function_args.get("query"), function_args.get("max_results", 5), api_keys.get("tavily"))
                            elif function_name == "get_weather":
                                tool_result = await get_weather(function_args.get("location"), function_args.get("units", "metric"), api_keys.get("openweather"))
                            else:
                                tool_result = "Unknown function call."

                            follow_up_prompt = f"Based on this tool result: {tool_result}\n\nPlease provide a comprehensive, Tony Stark-style answer to the original question: {prompt}"
                            follow_up_model = genai.GenerativeModel("gemini-1.5-flash")
                            follow_up_response = follow_up_model.generate_content(
                                history_for_model + [{"role": "user", "parts": [{"text": follow_up_prompt}]}],
                                stream=True
                            )
                            for follow_chunk in follow_up_response:
                                if follow_chunk.text:
                                    print(follow_chunk.text, end="", flush=True)
                                    full_llm_response_text += follow_chunk.text
                                    await murf_ws.send(json.dumps({"text": follow_chunk.text, "end": False}))
                        
                        elif hasattr(part, 'text') and part.text:
                            print(part.text, end="", flush=True)
                            full_llm_response_text += part.text
                            await murf_ws.send(json.dumps({"text": part.text, "end": False}))

            await murf_ws.send(json.dumps({"text": "", "end": True}))
            await murf_receiver_task
    except Exception as e:
        logger.error(f"Error during LLM stream: {str(e)}", exc_info=True)
    finally:
        if full_llm_response_text:
            chat_histories[session_id].append({"role": "user", "parts": [{"text": prompt}]})
            chat_histories[session_id].append({"role": "model", "parts": [{"text": full_llm_response_text}]})
        try:
            await client_websocket.send_text(json.dumps({"type": "audio_stream_end"}))
            await client_websocket.send_text(json.dumps({"type": "llm_response_text", "text": full_llm_response_text}))
        except Exception as e:
            logger.error(f"Failed to send final messages to client: {e}")
        print("\n--- End of LLM Stream ---\n")

# --- WebSocket endpoint now handles API key configuration ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session ID: {session_id}")

    api_keys = {}
    try:
        # First message must be the configuration with API keys
        config_message = await websocket.receive_json()
        if config_message.get("type") == "config":
            api_keys = config_message.get("keys", {})
            logger.info(f"Received API key configuration for session {session_id}")
            if not all(k in api_keys for k in ["gemini", "assemblyai", "murf"]):
                await websocket.send_text(json.dumps({"type": "error", "message": "Missing required API keys (Gemini, AssemblyAI, Murf)."}))
                return
        else:
            await websocket.send_text(json.dumps({"type": "error", "message": "First message must be a configuration object."}))
            return
    except Exception as e:
        logger.error(f"Error during config phase: {e}")
        await websocket.close()
        return

    audio_queue = asyncio.Queue()
    main_loop = asyncio.get_event_loop()
    state = {"last_transcript": "", "debounce_task": None}

    def on_turn(client, event: TurnEvent):
        if event.end_of_turn and event.transcript:
            logger.info(f"Transcription: {event.transcript}")
            state["last_transcript"] = event.transcript
            if state["debounce_task"]:
                state["debounce_task"].cancel()
            async def debounced_send():
                try:
                    await asyncio.sleep(1.2)
                    final_transcript = state["last_transcript"]
                    if not final_transcript: return
                    # Pass the API keys to the streaming function
                    asyncio.create_task(stream_llm_response_to_murf_and_client(final_transcript, websocket, session_id, api_keys))
                    await websocket.send_text(json.dumps({"type": "transcription", "text": final_transcript}))
                except asyncio.CancelledError: pass
                except Exception as e: logger.error(f"Error in debounced_send: {e}")
            state["debounce_task"] = asyncio.run_coroutine_threadsafe(debounced_send(), main_loop)

    def audio_generator():
        while True:
            try:
                data = asyncio.run_coroutine_threadsafe(audio_queue.get(), main_loop).result()
                if data is None: break
                yield data
            except Exception as e:
                logger.error(f"Error in audio_generator: {e}")
                break

    def run_transcriber(assemblyai_key: str):
        try:
            client = StreamingClient(StreamingClientOptions(api_key=assemblyai_key))
            client.on(StreamingEvents.Turn, on_turn)
            client.on(StreamingEvents.Error, lambda _, error: logger.error(f"A_AI error: {error}"))
            client.connect(StreamingParameters(sample_rate=16000, format_turns=True))
            client.stream(audio_generator())
        except Exception as e:
            logger.error(f"Error during AssemblyAI stream: {e}")

    async def receive_audio_task():
        try:
            while True:
                data = await websocket.receive_bytes()
                await audio_queue.put(data)
        except WebSocketDisconnect:
            logger.info("Client disconnected.")
            await audio_queue.put(None)

    try:
        assemblyai_api_key = api_keys.get("assemblyai")
        await asyncio.gather(asyncio.to_thread(run_transcriber, assemblyai_api_key), receive_audio_task())
    except Exception as e:
        logger.error(f"Error during concurrent execution: {e}")
    finally:
        logger.info(f"WebSocket endpoint for session {session_id} finished.")

# --- HTTP Endpoints (They will use keys from .env) ---

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ... (All other HTTP endpoints like /tts, /upload, /transcribe/file etc. remain unchanged)
# ... They will continue to use the API keys loaded from the .env file.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
