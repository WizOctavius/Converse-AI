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
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

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

# --- IMPROVED: Enhanced streaming function with better error handling ---
async def stream_llm_response_to_murf_and_client(prompt: str, client_websocket: WebSocket, session_id: str, api_keys: Dict[str, str]):
    gemini_api_key = api_keys.get("gemini")
    murf_api_key = api_keys.get("murf")

    if not gemini_api_key or not murf_api_key:
        logger.error("Critical API keys (Gemini, Murf) not provided for this session.")
        await client_websocket.send_text(json.dumps({"type": "error", "message": "Missing critical API keys"}))
        return

    logger.info(f"\n--- Starting LLM Stream for prompt: '{prompt}' ---")
    full_llm_response_text = ""
    murf_ws = None
    
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

        # IMPROVED: Better Murf WebSocket connection handling
        murf_ws_url = f"{MURF_WS_URL}?api-key={murf_api_key}&sample_rate=44100&channel_type=MONO&format=WAV"
        
        # Add connection timeout and retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                murf_ws = await websockets.connect(
                    murf_ws_url,
                    timeout=15.0,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                logger.info("Successfully connected to Murf WebSocket")
                break
            except (ConnectionClosed, InvalidStatusCode, asyncio.TimeoutError) as e:
                retry_count += 1
                logger.warning(f"Murf WebSocket connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(1)
                else:
                    raise e

        # Send voice configuration
        voice_config = {
            "voice_config": {
                "voiceId": "en-US-amara",
                "style": "Conversational",
                "speed": 1.0,
                "pitch": 0
            }
        }
        await murf_ws.send(json.dumps(voice_config))
        
        audio_chunk_count = 0
        
        # IMPROVED: Enhanced Murf audio receiver with better error handling
        async def receive_from_murf():
            nonlocal audio_chunk_count
            try:
                while True:
                    try:
                        response = await asyncio.wait_for(murf_ws.recv(), timeout=30.0)
                        data = json.loads(response)
                        
                        if "audio" in data and data["audio"]:
                            audio_chunk_count += 1
                            logger.debug(f"Received audio chunk {audio_chunk_count}")
                            await client_websocket.send_text(json.dumps({
                                "type": "audio_chunk", 
                                "audio_data": data["audio"]
                            }))
                        
                        if data.get("final", False):
                            logger.info(f"Murf stream completed. Total chunks: {audio_chunk_count}")
                            break
                            
                    except asyncio.TimeoutError:
                        logger.warning("Murf audio stream timed out")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode Murf response: {e}")
                        continue
                        
            except ConnectionClosed:
                logger.info("Murf WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error in Murf receiver: {e}")
            finally:
                logger.info(f"Murf receiver finished. Total audio chunks: {audio_chunk_count}")

        murf_receiver_task = asyncio.create_task(receive_from_murf())
        
        # Initialize LLM streaming
        model = genai.GenerativeModel("gemini-1.5-flash", tools=[{"function_declarations": get_function_declarations()}])
        response_stream = model.generate_content(history_for_model, stream=True)

        # Notify client that audio streaming is starting
        await client_websocket.send_text(json.dumps({"type": "audio_stream_start"}))

        # Process LLM stream
        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_name = part.function_call.name
                        function_args = dict(part.function_call.args)
                        logger.info(f"Function call: {function_name}({function_args})")
                        
                        if function_name == "search_web":
                            tool_result = await search_web(
                                function_args.get("query"), 
                                function_args.get("max_results", 5), 
                                api_keys.get("tavily", "")
                            )
                        elif function_name == "get_weather":
                            tool_result = await get_weather(
                                function_args.get("location"), 
                                function_args.get("units", "metric"), 
                                api_keys.get("openweather", "")
                            )
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
                                # Send text to Murf with error handling
                                try:
                                    await murf_ws.send(json.dumps({"text": follow_chunk.text, "end": False}))
                                except ConnectionClosed:
                                    logger.warning("Murf connection closed while sending text")
                                    break
                    
                    elif hasattr(part, 'text') and part.text:
                        print(part.text, end="", flush=True)
                        full_llm_response_text += part.text
                        # Send text to Murf with error handling
                        try:
                            await murf_ws.send(json.dumps({"text": part.text, "end": False}))
                        except ConnectionClosed:
                            logger.warning("Murf connection closed while sending text")
                            break

        # Signal end of text stream to Murf
        try:
            await murf_ws.send(json.dumps({"text": "", "end": True}))
            logger.info("Sent end signal to Murf")
        except ConnectionClosed:
            logger.warning("Murf connection closed before end signal")

        # Wait for all audio chunks to be received
        await murf_receiver_task
        
    except Exception as e:
        logger.error(f"Error during LLM stream: {str(e)}", exc_info=True)
        await client_websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Processing error: {str(e)}"
        }))
    finally:
        # Clean up Murf WebSocket connection
        if murf_ws:
            try:
                await murf_ws.close()
                logger.info("Murf WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Murf WebSocket: {e}")
        
        # Save chat history and send final messages
        if full_llm_response_text:
            chat_histories[session_id].append({"role": "user", "parts": [{"text": prompt}]})
            chat_histories[session_id].append({"role": "model", "parts": [{"text": full_llm_response_text}]})
        
        try:
            await client_websocket.send_text(json.dumps({"type": "audio_stream_end"}))
            await client_websocket.send_text(json.dumps({
                "type": "llm_response_text", 
                "text": full_llm_response_text
            }))
            logger.info("Sent final messages to client")
        except Exception as e:
            logger.error(f"Failed to send final messages to client: {e}")
        
        print("\n--- End of LLM Stream ---\n")

# --- IMPROVED: Enhanced WebSocket endpoint with better error handling ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session ID: {session_id}")

    api_keys = {}
    assemblyai_api_key = None
    
    try:
        # First message must be the configuration with API keys
        config_message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if config_message.get("type") == "config":
            api_keys = config_message.get("keys", {})
            assemblyai_api_key = api_keys.get("assemblyai")
            logger.info(f"Received API key configuration for session {session_id}")
            
            # Validate required API keys
            required_keys = ["gemini", "assemblyai", "murf"]
            missing_keys = [key for key in required_keys if not api_keys.get(key)]
            
            if missing_keys:
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": f"Missing required API keys: {', '.join(missing_keys)}"
                }))
                return
        else:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "First message must be a configuration object."
            }))
            return
    except asyncio.TimeoutError:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Configuration timeout. Please send API keys within 10 seconds."
        }))
        return
    except Exception as e:
        logger.error(f"Error during config phase: {e}")
        await websocket.close()
        return

    if not assemblyai_api_key:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "AssemblyAI API key is required for transcription."
        }))
        return

    # Set up transcription handling
    audio_queue = asyncio.Queue()
    main_loop = asyncio.get_event_loop()
    state = {"last_transcript": "", "debounce_task": None}

    def on_turn(client, event: TurnEvent):
        if event.end_of_turn and event.transcript:
            logger.info(f"Transcription: {event.transcript}")
            state["last_transcript"] = event.transcript
            
            # Cancel previous debounce task
            if state["debounce_task"]:
                state["debounce_task"].cancel()
            
            async def debounced_send():
                try:
                    await asyncio.sleep(1.2)  # Debounce delay
                    final_transcript = state["last_transcript"]
                    if not final_transcript: 
                        return
                    
                    # Send transcription to client
                    await websocket.send_text(json.dumps({
                        "type": "transcription", 
                        "text": final_transcript
                    }))
                    
                    # Start LLM processing
                    asyncio.create_task(stream_llm_response_to_murf_and_client(
                        final_transcript, websocket, session_id, api_keys
                    ))
                    
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in debounced_send: {e}")
                    
            state["debounce_task"] = asyncio.run_coroutine_threadsafe(debounced_send(), main_loop)

    def audio_generator():
        while True:
            try:
                data = asyncio.run_coroutine_threadsafe(audio_queue.get(), main_loop).result()
                if data is None:
                    break
                yield data
            except Exception as e:
                logger.error(f"Error in audio_generator: {e}")
                break

    def run_transcriber(assemblyai_key: str):
        try:
            client = StreamingClient(StreamingClientOptions(api_key=assemblyai_key))
            client.on(StreamingEvents.Turn, on_turn)
            client.on(StreamingEvents.Error, lambda _, error: logger.error(f"AssemblyAI error: {error}"))
            
            # Connect with enhanced parameters
            client.connect(StreamingParameters(
                sample_rate=16000, 
                format_turns=True,
                word_boost=["Tony", "Stark", "JARVIS", "AI", "assistant"]
            ))
            client.stream(audio_generator())
        except Exception as e:
            logger.error(f"Error during AssemblyAI stream: {e}")

    async def receive_audio_task():
        try:
            while True:
                data = await websocket.receive_bytes()
                await audio_queue.put(data)
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from session {session_id}")
            await audio_queue.put(None)
        except Exception as e:
            logger.error(f"Error receiving audio data: {e}")
            await audio_queue.put(None)

    try:
        # Run transcription and audio receiving concurrently
        await asyncio.gather(
            asyncio.to_thread(run_transcriber, assemblyai_api_key),
            receive_audio_task()
        )
    except Exception as e:
        logger.error(f"Error during concurrent execution: {e}")
    finally:
        logger.info(f"WebSocket endpoint for session {session_id} finished.")

# --- HTTP Endpoints (unchanged, using .env keys) ---
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint for deployment platforms
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Stark Voice Agent is operational"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
