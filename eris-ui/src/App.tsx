import { useState, useEffect, useRef } from 'react';
import Avatar3D from './components/Avatar3D';

// SET THIS TO TRUE ONCE YOU HAVE DOWNLOADED YOUR eris.glb FILE
const USE_3D_AVATAR = true;

type Message = { role: 'user' | 'eris'; text: string; id: string; reasoning?: string };
type Vitals = { turn_count: number; field_step_count: number; coherence: number; exchange: number; dCdX: number; regime: string; archetype: string };
type Voice = { id: string; name: string; engine: string };

// Web Speech API interfaces
const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [vitals, setVitals] = useState<Vitals | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const recognitionRef = useRef<any>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [llmReady, setLlmReady] = useState(false);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);
  
  const [volume, setVolume] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Fetch Voices with retry
    const fetchVoices = () => {
      fetch('/api/tts/voices')
        .then(res => res.json())
        .then(data => {
          if (data.voices) {
            setVoices(data.voices);
            if (data.voices.length > 0) {
              setSelectedVoice(data.voices[0].id);
            }
          }
        })
        .catch(err => {
          console.log("Backend not ready for TTS voices, retrying in 2s...");
          setTimeout(fetchVoices, 2000);
        });
    };
    fetchVoices();

    // Connect WebSocket for vitals
    const connectWs = () => {
      const ws = new WebSocket(`ws://${window.location.host}/ws`);
      ws.onmessage = (e) => setVitals(JSON.parse(e.data));
      ws.onclose = () => setTimeout(connectWs, 2000);
      wsRef.current = ws;
    };
    connectWs();
    
    // Poll LLM Status
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        setLlmReady(data.llm_ready);
        if (!data.llm_ready) {
          setTimeout(checkStatus, 3000);
        }
      } catch (e) {
        setTimeout(checkStatus, 3000);
      }
    };
    checkStatus();

    return () => wsRef.current?.close();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const playTTS = async (text: string) => {
    if (!selectedVoice) return;
    try {
      const res = await fetch('/api/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice_id: selectedVoice })
      });
      if (res.ok) {
        if (!audioCtxRef.current) {
          const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
          audioCtxRef.current = new AudioContext();
          analyserRef.current = audioCtxRef.current.createAnalyser();
          analyserRef.current.fftSize = 512;
          setAnalyser(analyserRef.current);
        }
        if (audioCtxRef.current.state === 'suspended') {
          audioCtxRef.current.resume();
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        activeAudioRef.current = audio;
        audio.volume = isMuted ? 0 : volume;
        
        const source = audioCtxRef.current.createMediaElementSource(audio);
        source.connect(analyserRef.current!);
        analyserRef.current!.connect(audioCtxRef.current.destination);

        audio.onplay = () => { setIsSpeaking(true); setIsPaused(false); };
        audio.onpause = () => setIsPaused(true);
        audio.onended = () => {
          setIsSpeaking(false);
          setIsPaused(false);
          source.disconnect();
          if (activeAudioRef.current === audio) activeAudioRef.current = null;
        };
        audio.play();
      }
    } catch (e) {
      console.error('TTS Error:', e);
    }
  };

  const toggleMic = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      if (!SpeechRecognition) return alert('Speech Recognition not supported in this browser.');
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        // We don't auto-send so you can edit, or we could auto-send here.
      };
      recognition.onend = () => setIsListening(false);
      recognition.start();
      setIsListening(true);
      recognitionRef.current = recognition;
    }
  };

  const handleStop = () => {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      setIsSpeaking(false);
      setIsPaused(false);
    }
  };

  const handlePauseToggle = () => {
    if (activeAudioRef.current) {
      if (activeAudioRef.current.paused) activeAudioRef.current.play();
      else activeAudioRef.current.pause();
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = parseFloat(e.target.value);
    setVolume(vol);
    if (activeAudioRef.current) {
      activeAudioRef.current.volume = isMuted ? 0 : vol;
    }
  };

  const toggleMute = () => {
    const newMuted = !isMuted;
    setIsMuted(newMuted);
    if (activeAudioRef.current) {
      activeAudioRef.current.volume = newMuted ? 0 : volume;
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isProcessing) return;
    
    handleStop(); // Stop any currently playing audio when sending a new message

    const text = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text, id: Date.now().toString() }]);
    setIsProcessing(true);

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await res.json();
      
      setMessages(prev => [...prev, { role: 'eris', text: data.response, reasoning: data.reasoning, id: Date.now().toString() }]);
      playTTS(data.response);
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <>
      <div className="glass-panel chat-container">
        <div className="chat-header">
          <h1>Eris Echo v4</h1>
          <div style={{ color: isProcessing ? 'var(--accent-primary)' : (llmReady ? 'var(--text-muted)' : 'orange') }}>
            {isProcessing ? 'Processing...' : (llmReady ? 'Online' : 'Loading GPT-OSS into GPU...')}
          </div>
        </div>
        
        <div className="messages">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '2rem' }}>
              System online. FRACTAL PDE engine running.
            </div>
          )}
          {messages.map(msg => (
            <div key={msg.id} className={`message ${msg.role}`}>
              {msg.reasoning && (
                  <details className="reasoning-dropdown">
                    <summary>Eris's Cognitive Process</summary>
                    <pre>{msg.reasoning}</pre>
                  </details>
              )}
              {msg.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <button 
            className={`mic-button ${isListening ? 'listening' : ''}`} 
            onClick={toggleMic} 
            disabled={isProcessing || !llmReady}
            title="Toggle Microphone"
          >
            {isListening ? '🛑' : '🎤'}
          </button>
          <input 
            type="text" 
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
            placeholder={llmReady ? "Speak with Eris..." : "Waiting for GPT-OSS to load..."}
            disabled={isProcessing || !llmReady}
          />
          <button onClick={sendMessage} disabled={isProcessing || !llmReady}>
            Send
          </button>
        </div>
      </div>

      <div className="sidebar">
        <div className="glass-panel panel avatar-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center', alignItems: 'center' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%', alignItems: 'center' }}>
            <img src="/eris_avatar.png" alt="Eris Avatar" className={`eris-avatar ${isSpeaking ? 'speaking' : ''}`} style={{ maxHeight: '150px', objectFit: 'contain' }} />
            {USE_3D_AVATAR && (
              <div style={{ width: '100%' }}>
                <Avatar3D isTalking={isSpeaking} vitals={vitals || { dCdX: 0.0, coherence: 0.0 }} analyser={analyser} />
              </div>
            )}
            <button 
              onClick={() => window.open('/visualizer.html', '_blank', 'width=840,height=600,menubar=no,toolbar=no')} 
              style={{ padding: '0.5rem 1rem', background: 'var(--accent-primary)', border: 'none', borderRadius: '4px', color: 'white', cursor: 'pointer', marginTop: '1rem', width: '100%', fontFamily: 'var(--mono-font)' }}
            >
              Open Live Field Visualizer
            </button>
          </div>
        </div>

        <div className="glass-panel panel">
          <h3>Cognitive Vitals</h3>
          <div className="vital-row">
            <span className="vital-label">Archetype</span>
            <span className="vital-value">{vitals?.archetype || '---'}</span>
          </div>
          <div className="vital-row">
            <span className="vital-label">Regime</span>
            <span className="vital-value">{vitals?.regime || '---'}</span>
          </div>
          <div className="vital-row">
            <span className="vital-label">Dissonance (dC/dX)</span>
            <span className="vital-value">{vitals?.dCdX?.toFixed(4) || '0.0000'}</span>
          </div>
          <div className="vital-row">
            <span className="vital-label">Coherence</span>
            <span className="vital-value">{vitals?.coherence?.toFixed(3) || '0.000'}</span>
          </div>
          <div className="vital-row">
            <span className="vital-label">Turns</span>
            <span className="vital-value">{vitals?.turn_count || 0}</span>
          </div>
        </div>

        <div className="glass-panel panel">
          <h3>Voice Settings</h3>
          <select 
            value={selectedVoice}
            onChange={e => setSelectedVoice(e.target.value)}
          >
            {voices.map(v => (
              <option key={v.id} value={v.id}>
                {v.name} ({v.engine})
              </option>
            ))}
          </select>
          <button 
            style={{ padding: '0.8rem', marginTop: '0.5rem' }}
            onClick={() => playTTS("I am Eris. The current cognitive state is stable.")}
          >
            Test Voice
          </button>
          
          <div className="audio-controls" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
              <button onClick={handlePauseToggle} disabled={!isSpeaking && !isPaused} style={{ flex: 1, padding: '0.5rem' }}>
                {isPaused ? '▶️ Play' : '⏸ Pause'}
              </button>
              <button onClick={handleStop} disabled={!isSpeaking && !isPaused} style={{ flex: 1, padding: '0.5rem', background: '#5a3b5a' }}>
                ⏹ Stop
              </button>
              <button onClick={toggleMute} style={{ padding: '0.5rem' }}>
                {isMuted ? '🔇' : '🔊'}
              </button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Volume</span>
              <input 
                type="range" min="0" max="1" step="0.05" 
                value={volume} onChange={handleVolumeChange}
                style={{ flex: 1 }}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
