document.addEventListener('DOMContentLoaded', () => {
    // --- API Key Management & UI Elements ---
    let apiKeys = {
        gemini: null,
        assemblyai: null,
        murf: null,
        tavily: null,
        openweather: null,
    };
    const settingsModal = document.getElementById('settingsModal');
    const settingsBtn = document.getElementById('settingsBtn');
    const closeSettingsBtn = document.getElementById('closeSettingsBtn');
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    const geminiApiKeyInput = document.getElementById('geminiApiKey');
    const assemblyaiApiKeyInput = document.getElementById('assemblyaiApiKey');
    const murfApiKeyInput = document.getElementById('murfApiKey');
    const tavilyApiKeyInput = document.getElementById('tavilyApiKey');
    const openweatherApiKeyInput = document.getElementById('openweatherApiKey');

    // --- Session management ---
    let isRecording = false;
    let sessionId = null;

    // --- IMPROVED AudioPlayer Class ---
    class AudioPlayer {
        constructor() {
            this.audioContext = null;
            this.rawAudioChunks = [];
            this.isPlaying = false;
            this.audioQueue = [];
        }

        async start() {
            try {
                // Create new AudioContext with proper initialization
                if (!this.audioContext || this.audioContext.state === 'closed') {
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                // Ensure AudioContext is running
                if (this.audioContext.state === 'suspended') {
                    await this.audioContext.resume();
                }
                
                this.rawAudioChunks = [];
                this.audioQueue = [];
                this.isPlaying = false;
                console.log("AudioPlayer started, AudioContext state:", this.audioContext.state);
            } catch (error) {
                console.error("Failed to start AudioPlayer:", error);
                updateStatus("Audio initialization failed", true);
            }
        }

        async stop() {
            this.isPlaying = false;
            this.rawAudioChunks = [];
            this.audioQueue = [];
            // Don't close the AudioContext, just suspend it for reuse
            if (this.audioContext && this.audioContext.state === 'running') {
                await this.audioContext.suspend();
            }
        }

        queueChunk(arrayBuffer) {
            if (!arrayBuffer || arrayBuffer.byteLength === 0) {
                console.warn("Received empty audio chunk");
                return;
            }
            this.rawAudioChunks.push(arrayBuffer);
            console.log(`Queued audio chunk: ${arrayBuffer.byteLength} bytes, total chunks: ${this.rawAudioChunks.length}`);
        }

        async play() {
            if (this.rawAudioChunks.length === 0) {
                console.warn("No audio chunks to play");
                return;
            }

            if (this.isPlaying) {
                console.log("Already playing audio, queuing for later");
                return;
            }

            console.log(`Starting playback of ${this.rawAudioChunks.length} audio chunks`);
            this.isPlaying = true;

            try {
                // Ensure AudioContext is ready
                if (!this.audioContext || this.audioContext.state === 'closed') {
                    await this.start();
                }

                if (this.audioContext.state === 'suspended') {
                    await this.audioContext.resume();
                }

                // Combine all chunks
                const totalLength = this.rawAudioChunks.reduce((acc, chunk) => acc + chunk.byteLength, 0);
                const combined = new Uint8Array(totalLength);
                let offset = 0;
                
                for (const chunk of this.rawAudioChunks) {
                    combined.set(new Uint8Array(chunk), offset);
                    offset += chunk.byteLength;
                }

                console.log(`Combined audio data: ${combined.length} bytes`);

                // Create WAV buffer and decode
                const wavBuffer = this.createWavBuffer(combined);
                const audioBuffer = await this.audioContext.decodeAudioData(wavBuffer);
                
                console.log(`Decoded audio: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz`);

                // Play the audio
                const source = this.audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this.audioContext.destination);

                // Handle playback completion
                source.onended = () => {
                    console.log("Audio playback completed");
                    this.isPlaying = false;
                    this.rawAudioChunks = [];
                    updateStatus("Ready for command.", false);
                };

                source.start(0);
                updateStatus("Playing audio response...", false);

            } catch (error) {
                console.error("Failed to decode and play audio:", error);
                updateStatus("Error playing audio response", true);
                this.isPlaying = false;
                this.rawAudioChunks = [];
            }
        }

        createWavBuffer(pcmData) {
            const sampleRate = 44100;
            const numChannels = 1;
            const bytesPerSample = 2;
            const blockAlign = numChannels * bytesPerSample;
            const byteRate = sampleRate * blockAlign;
            const dataSize = pcmData.length;
            
            const buffer = new ArrayBuffer(44 + dataSize);
            const view = new DataView(buffer);
            
            // WAV header
            this.writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + dataSize, true);
            this.writeString(view, 8, 'WAVE');
            this.writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, byteRate, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bytesPerSample * 8, true);
            this.writeString(view, 36, 'data');
            view.setUint32(40, dataSize, true);
            
            // Audio data
            new Uint8Array(buffer, 44).set(pcmData);
            return buffer;
        }

        writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }
    }
    const audioPlayer = new AudioPlayer();

    // --- REVAMPED UI Elements ---
    const mainRecordBtn = document.getElementById('mainRecordBtn');
    const arcReactorIcon = document.getElementById('arcReactorIcon');
    const arcCore = document.getElementById('arc-core');
    const chatStatus = document.getElementById('chatStatus');
    const chatVisualizer = document.getElementById('chatVisualizer');
    const chatCanvasCtx = chatVisualizer.getContext('2d');
    const newSessionBtn = document.getElementById('newSessionBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const chatHistory = document.getElementById('chatHistory');

    // WebSocket and microphone streaming variables
    let scriptNode;
    let mediaStreamSource;
    let websocket;
    const TARGET_SAMPLE_RATE = 16000;
    let chatAnimationId;

    // --- API Key Management (Unchanged) ---
    function saveApiKeys() {
        apiKeys.gemini = geminiApiKeyInput.value.trim();
        apiKeys.assemblyai = assemblyaiApiKeyInput.value.trim();
        apiKeys.murf = murfApiKeyInput.value.trim();
        apiKeys.tavily = tavilyApiKeyInput.value.trim();
        apiKeys.openweather = openweatherApiKeyInput.value.trim();
        localStorage.setItem('userApiKeys', JSON.stringify(apiKeys));
        alert('API Keys saved successfully!');
        settingsModal.classList.add('hidden');
    }

    function loadApiKeys() {
        const storedKeys = localStorage.getItem('userApiKeys');
        if (storedKeys) {
            apiKeys = JSON.parse(storedKeys);
            geminiApiKeyInput.value = apiKeys.gemini || '';
            assemblyaiApiKeyInput.value = apiKeys.assemblyai || '';
            murfApiKeyInput.value = apiKeys.murf || '';
            tavilyApiKeyInput.value = apiKeys.tavily || '';
            openweatherApiKeyInput.value = apiKeys.openweather || '';
        }
    }

    function checkRequiredKeys() {
        if (!apiKeys.gemini || !apiKeys.assemblyai || !apiKeys.murf) {
            updateStatus("API Keys required. Open settings to add them.", true);
            return false;
        }
        return true;
    }

    settingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
    closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
    saveSettingsBtn.addEventListener('click', saveApiKeys);

    // --- Session Initialization ---
    function initializeSession() {
        sessionId = crypto.randomUUID();
        console.log(`New session started: ${sessionId}`);
        chatHistory.innerHTML = `<div class="initial-message text-gray-400 italic text-center flex items-center justify-center h-full flex-grow"><div class="text-lg">Awaiting audio input...</div></div>`;
        audioPlayer.stop();
        if (!checkRequiredKeys()) {
            setTimeout(() => settingsModal.classList.remove('hidden'), 500);
        } else {
            updateStatus("Session initialized. Ready for command.", false);
        }
    }

    function base64ToArrayBuffer(base64String) {
        try {
            const binaryString = window.atob(base64String);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return bytes.buffer;
        } catch (error) {
            console.error("Failed to decode base64 audio:", error);
            return null;
        }
    }

    // --- UI State Management Functions ---
    function updateRecordButton(state) {
        switch (state) {
            case 'idle':
                mainRecordBtn.classList.remove('recording-glow');
                arcCore.setAttribute('fill', 'var(--stark-light-blue)');
                arcReactorIcon.classList.add('arc-reactor-pulse');
                break;
            case 'recording':
                mainRecordBtn.classList.add('recording-glow');
                arcCore.setAttribute('fill', 'var(--stark-red)');
                arcReactorIcon.classList.remove('arc-reactor-pulse');
                break;
            case 'processing':
                mainRecordBtn.classList.remove('recording-glow');
                arcCore.setAttribute('fill', 'var(--stark-gold)');
                arcReactorIcon.classList.remove('arc-reactor-pulse');
                break;
        }
    }

    function updateStatus(message, isError = false) {
        console.log("Status update:", message);
        chatStatus.textContent = message;
        chatStatus.className = `text-lg font-semibold ${isError ? 'text-red-400' : 'text-[var(--stark-light-blue)]'}`;
        chatStatus.classList.remove('hidden');
    }

    function addMessageToHistory(text, role = 'user') {
        const initialMessage = chatHistory.querySelector('.initial-message');
        if (initialMessage) {
            initialMessage.remove();
        }
        const messageDiv = document.createElement('div');
        if (role === 'user') {
            messageDiv.className = 'chat-message mb-4 flex justify-end animate-fadeInUp';
            messageDiv.innerHTML = `<div class="bg-[var(--stark-light-blue)] text-black rounded-lg rounded-br-none py-2 px-4 max-w-sm shadow-md font-medium">${text}</div>`;
        } else {
            messageDiv.className = 'chat-message mb-4 flex justify-start animate-fadeInUp';
            messageDiv.innerHTML = `<div class="assistant-bubble bg-gray-700 text-white rounded-lg rounded-bl-none py-2 px-4 max-w-sm shadow-md">${text}</div>`;
        }
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function updateAssistantMessage(text) {
        const assistantBubbles = chatHistory.querySelectorAll('.assistant-bubble');
        if (assistantBubbles.length > 0) {
            const lastBubble = assistantBubbles[assistantBubbles.length - 1];
            lastBubble.textContent = text;
        }
    }

    // --- IMPROVED WebSocket and Audio Handling ---
    function handleAudioStreamStart() {
        console.log("Audio streaming started from server");
        updateStatus("Receiving audio stream...", false);
        addMessageToHistory("...", 'assistant');
        // Initialize audio player for incoming stream
        audioPlayer.start();
    }

    function handleAudioChunk(chunkData) {
        const { audio_data } = chunkData;
        if (!audio_data) {
            console.warn("Received audio chunk without data");
            return;
        }
        
        const arrayBuffer = base64ToArrayBuffer(audio_data);
        if (arrayBuffer) {
            audioPlayer.queueChunk(arrayBuffer);
        }
    }

    async function handleAudioStreamEnd() {
        console.log("Audio stream completed from server");
        updateStatus("Audio stream complete. Processing...", false);
        // Small delay to ensure all chunks are received
        setTimeout(() => {
            audioPlayer.play();
        }, 100);
    }

    mainRecordBtn.addEventListener('click', () => {
        if (!isRecording) {
            startStreaming();
        } else {
            stopStreaming();
        }
    });

    clearHistoryBtn.addEventListener('click', () => {
        chatHistory.innerHTML = `<div class="initial-message text-gray-400 italic text-center flex items-center justify-center h-full flex-grow"><div class="text-lg">Log Cleared.</div></div>`;
        audioPlayer.stop();
    });

    newSessionBtn.addEventListener('click', initializeSession);

    async function startStreaming() {
        if (!checkRequiredKeys()) {
            settingsModal.classList.remove('hidden');
            return;
        }

        try {
            // Initialize audio player first
            await audioPlayer.start();

            // Get microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    channelCount: 1,
                    sampleRate: 44100,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            // Determine WebSocket URL based on current protocol
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
            console.log(`Connecting to WebSocket: ${wsUrl}`);

            websocket = new WebSocket(wsUrl);

            websocket.onopen = () => {
                console.log("WebSocket connected. Sending API keys configuration.");
                websocket.send(JSON.stringify({ type: 'config', keys: apiKeys }));
                updateStatus('Connected. Recording...', false);
                isRecording = true;
                updateRecordButton('recording');

                // Set up audio processing
                const userAudioContext = new (window.AudioContext || window.webkitAudioContext)();
                mediaStreamSource = userAudioContext.createMediaStreamSource(stream);
                const bufferSize = 4096;
                scriptNode = userAudioContext.createScriptProcessor(bufferSize, 1, 1);
                
                scriptNode.onaudioprocess = (audioProcessingEvent) => {
                    if (!isRecording || websocket.readyState !== WebSocket.OPEN) return;
                    
                    const pcmData = audioProcessingEvent.inputBuffer.getChannelData(0);
                    const resampledData = resampleBuffer(pcmData, userAudioContext.sampleRate, TARGET_SAMPLE_RATE);
                    const pcm16Data = convertTo16BitPCM(resampledData);
                    
                    try {
                        websocket.send(pcm16Data.buffer);
                    } catch (error) {
                        console.error("Failed to send audio data:", error);
                    }
                };

                mediaStreamSource.connect(scriptNode);
                scriptNode.connect(userAudioContext.destination);
                startChatVisualizer(stream);
            };

            websocket.onmessage = (event) => {
                if (typeof event.data !== 'string' || event.data.trim() === '') return;
                
                let message;
                try {
                    message = JSON.parse(event.data);
                } catch (e) {
                    console.error("Failed to parse WebSocket message:", event.data, e);
                    return;
                }

                console.log("Received WebSocket message:", message.type);

                if (message.type === 'error') {
                    console.error("Server error:", message.message);
                    updateStatus(`Server Error: ${message.message}`, true);
                    stopStreaming();
                    return;
                }

                switch (message.type) {
                    case 'transcription':
                        if (message.text && message.text.trim() !== '') {
                            console.log("Transcription received:", message.text);
                            addMessageToHistory(message.text, 'user');
                        }
                        break;
                    case 'audio_stream_start':
                        handleAudioStreamStart();
                        break;
                    case 'audio_chunk':
                        handleAudioChunk(message);
                        break;
                    case 'audio_stream_end':
                        handleAudioStreamEnd();
                        break;
                    case 'llm_response_text':
                        console.log("LLM response received:", message.text);
                        updateAssistantMessage(message.text);
                        break;
                    default:
                        console.warn("Unknown message type:", message.type);
                }
            };

            websocket.onclose = (event) => {
                console.log("WebSocket closed:", event.code, event.reason);
                updateStatus('Connection closed.', false);
                if (isRecording) stopStreaming();
            };

            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateStatus('Connection error occurred.', true);
                if (isRecording) stopStreaming();
            };

        } catch (error) {
            console.error("Failed to start streaming:", error);
            updateStatus('Failed to access microphone or connect.', true);
        }
    }

    function stopStreaming() {
        if (!isRecording) return;
        
        console.log("Stopping streaming...");
        isRecording = false;
        updateRecordButton('processing');
        
        // Clean up WebSocket
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.close();
        }

        // Clean up audio processing
        if (scriptNode) {
            scriptNode.disconnect();
            scriptNode = null;
        }
        
        if (mediaStreamSource) {
            mediaStreamSource.mediaStream.getTracks().forEach(track => track.stop());
            mediaStreamSource.disconnect();
            mediaStreamSource = null;
        }
        
        stopChatVisualizer();

        setTimeout(() => {
            updateRecordButton('idle');
            updateStatus("Ready for command.");
        }, 2000);
    }

    // --- Audio Processing Utilities (Unchanged) ---
    function resampleBuffer(inputBuffer, fromSampleRate, toSampleRate) {
        if (fromSampleRate === toSampleRate) return inputBuffer;
        
        const sampleRateRatio = fromSampleRate / toSampleRate;
        const newLength = Math.round(inputBuffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
            const index = i * sampleRateRatio;
            const indexFloor = Math.floor(index);
            const indexCeil = Math.ceil(index);
            
            if (indexCeil >= inputBuffer.length) {
                result[i] = inputBuffer[inputBuffer.length - 1];
            } else {
                const weight = index - indexFloor;
                result[i] = inputBuffer[indexFloor] * (1 - weight) + inputBuffer[indexCeil] * weight;
            }
        }
        return result;
    }

    function convertTo16BitPCM(input) {
        const output = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return output;
    }

    // --- Visualizer Functions (Unchanged) ---
    function startChatVisualizer(stream) {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioCtx.createAnalyser();
        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        chatVisualizer.classList.remove('hidden');

        function drawChat() {
            chatAnimationId = requestAnimationFrame(drawChat);
            analyser.getByteFrequencyData(dataArray);
            chatCanvasCtx.clearRect(0, 0, chatVisualizer.width, chatVisualizer.height);
            const barWidth = (chatVisualizer.width / bufferLength) * 1.5;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const barHeight = dataArray[i] / 2.8;
                const gradient = chatCanvasCtx.createLinearGradient(0, chatVisualizer.height, 0, chatVisualizer.height - barHeight);
                gradient.addColorStop(0, 'rgba(211, 47, 47, 0.2)');
                gradient.addColorStop(0.5, 'rgba(255, 193, 7, 0.5)');
                gradient.addColorStop(1, 'rgba(100, 255, 218, 1)');
                chatCanvasCtx.fillStyle = gradient;
                chatCanvasCtx.fillRect(x, chatVisualizer.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        }
        drawChat();
    }

    function stopChatVisualizer() {
        if (chatAnimationId) cancelAnimationFrame(chatAnimationId);
        if (chatVisualizer) {
            const ctx = chatVisualizer.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, chatVisualizer.width, chatVisualizer.height);
            }
            chatVisualizer.classList.add('hidden');
        }
    }

    // --- Initialize on load ---
    loadApiKeys();
    initializeSession();
    updateRecordButton('idle');
});
