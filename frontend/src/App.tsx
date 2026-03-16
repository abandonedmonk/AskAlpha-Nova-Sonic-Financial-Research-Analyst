import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, MessageSquare, Database, Activity, Send, Terminal, ChevronLeft, File, Clock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Components ---

const Panel = ({ title, children, className, icon: Icon }: { title: string, children: React.ReactNode, className?: string, icon?: any }) => (
  <div className={cn("bg-[#141414] border border-[#262626] rounded-xl flex flex-col overflow-hidden h-full shadow-2xl", className)}>
    <div className="px-4 py-3 border-b border-[#262626] flex items-center justify-between bg-[#1a1a1a]">
      <div className="flex items-center gap-2">
        {Icon && <Icon size={16} className="text-blue-400" />}
        <h2 className="text-[11px] font-bold text-gray-400 tracking-[0.2em] uppercase">{title}</h2>
      </div>
      <div className="flex gap-1">
        <div className="w-2 h-2 rounded-full bg-red-500/20" />
        <div className="w-2 h-2 rounded-full bg-yellow-500/20" />
        <div className="w-2 h-2 rounded-full bg-green-500/20" />
      </div>
    </div>
    <div className="flex-1 overflow-y-auto p-4 custom-scrollbar bg-[#0d0d0d]/50">
      {children}
    </div>
  </div>
);

const MessageBubble = ({ text, isUser }: { text: string, isUser: boolean }) => (
  <motion.div 
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className={cn(
      "mb-4 p-3 rounded-lg border border-dashed text-sm leading-relaxed",
      isUser 
        ? "border-blue-500/30 bg-blue-500/5 ml-8 text-blue-100/90" 
        : "border-gray-700 bg-gray-800/20 mr-8 text-gray-300"
    )}
  >
    {text}
  </motion.div>
);

const LogEntry = ({ text, type = 'info' }: { text: string, type?: 'info' | 'tool' | 'error' }) => (
  <motion.div 
    initial={{ opacity: 0, x: 10 }}
    animate={{ opacity: 1, x: 0 }}
    className={cn(
      "mb-2 p-2 rounded border border-dashed text-[10px] font-mono flex gap-2 items-start",
      type === 'tool' ? "border-purple-500/30 bg-purple-500/5 text-purple-300" :
      type === 'error' ? "border-red-500/30 bg-red-500/5 text-red-300" :
      "border-gray-800 bg-gray-900/30 text-gray-500"
    )}
  >
    <span className="opacity-50 mt-0.5">[{new Date().toLocaleTimeString([], {hour12: false})}]</span>
    <span className="flex-1">{text}</span>
  </motion.div>
);

const VaultFileItem = ({ filename, modified, onClick }: { filename: string, modified: number, onClick: () => void }) => (
  <motion.div
    initial={{ opacity: 0, x: -10 }}
    animate={{ opacity: 1, x: 0 }}
    whileHover={{ x: 4 }}
    onClick={onClick}
    className="flex items-center gap-3 p-3 rounded-lg border border-dashed border-blue-500/20 bg-blue-500/5 hover:bg-blue-500/10 cursor-pointer transition-all group"
  >
    <File size={14} className="text-blue-400/60 group-hover:text-blue-400 transition-colors" />
    <div className="flex-1 min-w-0">
      <p className="text-xs font-mono text-gray-300 truncate group-hover:text-blue-200">{filename}</p>
      <div className="flex items-center gap-1 mt-1 text-[8px] text-gray-500">
        <Clock size={10} />
        <span>{new Date(modified * 1000).toLocaleDateString()}</span>
      </div>
    </div>
    <div className="text-gray-600 group-hover:text-blue-500 transition-colors">→</div>
  </motion.div>
);

// Formatted Result Display Component
const FormattedResult = ({ data, toolName }: { data: any; toolName: string }) => {
  if (toolName === 'execute_quantitative_model') {
    // Monte Carlo simulation results
    const monte = data?.monte_carlo || data;
    return (
      <div className="space-y-3 text-xs">
        <div className="flex items-center justify-between border-b border-gray-700 pb-2">
          <span className="text-gray-400 font-bold uppercase tracking-wider">Monte Carlo Simulation</span>
        </div>
        <div className="space-y-2">
          {monte.ticker && (
            <div className="flex justify-between items-start">
              <span className="text-gray-500">Ticker:</span>
              <span className="text-blue-300 font-mono font-bold">{monte.ticker}</span>
            </div>
          )}
          {monte.current_price && (
            <div className="flex justify-between items-start">
              <span className="text-gray-500">Price:</span>
              <span className="text-green-400 font-mono">${monte.current_price.toFixed(2)}</span>
            </div>
          )}
          {monte.volatility && (
            <div className="flex justify-between items-start">
              <span className="text-gray-500">Volatility:</span>
              <span className="text-yellow-300 font-mono">{(monte.volatility * 100).toFixed(2)}%</span>
            </div>
          )}
          {monte.percentiles && (
            <div className="border-t border-gray-700 pt-2 mt-2">
              <div className="text-gray-400 font-bold mb-1">Percentiles (End Price):</div>
              <div className="grid grid-cols-3 gap-2 pl-2">
                {monte.percentiles.p10 && (
                  <div><span className="text-gray-500">p10:</span> <span className="text-orange-300">${monte.percentiles.p10.toFixed(2)}</span></div>
                )}
                {monte.percentiles.p50 && (
                  <div><span className="text-gray-500">p50:</span> <span className="text-blue-300">${monte.percentiles.p50.toFixed(2)}</span></div>
                )}
                {monte.percentiles.p90 && (
                  <div><span className="text-gray-500">p90:</span> <span className="text-green-300">${monte.percentiles.p90.toFixed(2)}</span></div>
                )}
              </div>
            </div>
          )}
          {monte.num_simulations && (
            <div className="border-t border-gray-700 pt-2 mt-2">
              <div className="flex justify-between items-start">
                <span className="text-gray-500">Simulations:</span>
                <span className="text-purple-300 font-mono">{monte.num_simulations.toLocaleString()}</span>
              </div>
            </div>
          )}
          {monte.days && (
            <div className="flex justify-between items-start">
              <span className="text-gray-500">Period:</span>
              <span className="text-purple-300 font-mono">{monte.days} days</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (toolName === 'query_live_market_data') {
    // Market snapshot data
    return (
      <div className="space-y-3 text-xs">
        <div className="flex items-center justify-between border-b border-gray-700 pb-2">
          <span className="text-gray-400 font-bold uppercase tracking-wider">Market Snapshot</span>
        </div>
        <div className="space-y-2">
          {data?.ticker && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">COMPANY (Ticker)</span>
              <span className="text-blue-400 font-mono font-bold">{data.ticker}</span>
            </div>
          )}
          {data?.price !== undefined && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">PRICE</span>
              <span className="text-green-400 font-mono">${data.price.toFixed(2)}</span>
            </div>
          )}
          {data?.volume !== undefined && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">VOLUME</span>
              <span className="text-yellow-400 font-mono">{(data.volume / 1000000).toFixed(2)} M</span>
            </div>
          )}
          {data?.change_percent !== undefined && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">CHANGE:</span>
              <span className={data.change_percent >= 0 ? "text-green-400 font-mono" : "text-red-400 font-mono"}>
                {data.change_percent >= 0 ? '+' : ''}{data.change_percent.toFixed(2)}%
              </span>
            </div>
          )}
          {data?.high_52w && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">52W HIGH:</span>
              <span className="text-blue-400 font-mono">${data.high_52w.toFixed(2)}</span>
            </div>
          )}
          {data?.low_52w && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">52W LOW:</span>
              <span className="text-blue-400 font-mono">${data.low_52w.toFixed(2)}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (toolName === 'analyze_sec_filings_rag') {
    // SEC filing analysis
    return (
      <div className="space-y-3 text-xs">
        <div className="flex items-center justify-between border-b border-gray-700 pb-2">
          <span className="text-gray-500 font-bold uppercase tracking-wider">SEC Filing Analysis</span>
        </div>
        <div className="space-y-2">
          {data?.filing_type && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">TYPE:</span>
              <span className="text-blue-400 font-mono">{data.filing_type}</span>
            </div>
          )}
          {data?.company && (
            <div className="flex justify-between items-start">
              <span className="text-gray-400">COMPANY:</span>
              <span className="text-blue-400 font-mono">{data.company}</span>
            </div>
          )}
          {data?.summary && (
            <div className="border-t border-gray-700 pt-2 mt-2">
              <div className="text-gray-400 font-bold mb-1">SUMMARY:</div>
              <p className="text-gray-400 pl-2 leading-relaxed">{data.summary}</p>
            </div>
          )}
          {data?.key_metrics && (
            <div className="border-t border-gray-700 pt-2 mt-2">
              <div className="text-gray-400 font-bold mb-1">KEY METRICS:</div>
              <div className="pl-2 space-y-1">
                {Object.entries(data.key_metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-500">{key}:</span>
                    <span className="text-blue-400">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Fallback for unknown tool types - show formatted JSON
  return (
    <div className="space-y-3 text-xs">
      <div className="flex items-center justify-between border-b border-gray-700 pb-2">
        <span className="text-gray-400 font-bold uppercase tracking-wider">{toolName}</span>
      </div>
      <pre className="text-green-400/90 leading-relaxed overflow-x-auto whitespace-pre-wrap text-[9px]">
        {typeof data === 'string' ? data : JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
};

const VoiceVisualizer = ({ isActive }: { isActive: boolean }) => {
  return (
    <div className="relative w-80 h-80 flex items-center justify-center">
      {/* Dynamic Glow Layers */}
      <motion.div 
        animate={{ 
          scale: isActive ? [1, 1.2, 1] : 1,
          opacity: isActive ? [0.2, 0.4, 0.2] : 0.15
        }}
        transition={{ repeat: Infinity, duration: 3 }}
        className="absolute inset-0 bg-blue-600/20 rounded-full blur-[80px]"
      />
      <motion.div 
        animate={{ 
          scale: isActive ? [1, 1.1, 1] : 1,
          opacity: isActive ? [0.3, 0.5, 0.3] : 0.2
        }}
        transition={{ repeat: Infinity, duration: 2, delay: 0.5 }}
        className="absolute inset-10 bg-purple-600/10 rounded-full blur-[60px]"
      />
      
      {/* The Central Sphere */}
      <div className="relative w-56 h-56 rounded-full overflow-hidden border border-white/5 bg-gradient-to-b from-[#1a1a1a] to-black flex items-center justify-center shadow-[0_0_60px_rgba(0,0,0,1)]">
        {/* Animated Grid / Pattern inside */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#3b82f6_1px,transparent_1px)] [background-size:16px_16px]" />
        
        {/* Smooth Wave Visualization */}
        <div className="absolute inset-0 flex items-center justify-center">
          <AnimatePresence mode="wait">
            {isActive ? (
              <svg
                key="wave-svg"
                viewBox="0 0 200 120"
                className="w-40 h-24"
                preserveAspectRatio="none"
              >
                <defs>
                  <linearGradient id="waveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="rgba(59, 130, 246, 0.8)" />
                    <stop offset="100%" stopColor="rgba(147, 51, 234, 0.4)" />
                  </linearGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                      <feMergeNode in="coloredBlur"/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                </defs>
                
                {/* Multiple stacked waves for layered effect */}
                {[0, 1, 2].map((i) => (
                  <motion.path
                    key={`wave-${i}`}
                    fill="none"
                    stroke="url(#waveGradient)"
                    strokeWidth="3"
                    strokeLinecap="round"
                    filter="url(#glow)"
                    opacity={isActive ? 0.6 - i * 0.15 : 0.2}
                    initial={{ d: "M 0,60 Q 25,40 50,60 T 100,60 T 150,60 T 200,60" }}
                    animate={{
                      d: [
                        "M 0,60 Q 25,40 50,60 T 100,60 T 150,60 T 200,60",
                        "M 0,60 Q 25,45 50,55 T 100,60 T 150,65 T 200,60",
                        "M 0,60 Q 25,50 50,50 T 100,60 T 150,70 T 200,60",
                        "M 0,60 Q 25,45 50,55 T 100,60 T 150,65 T 200,60",
                        "M 0,60 Q 25,40 50,60 T 100,60 T 150,60 T 200,60",
                      ]
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 2 + i * 0.3,
                      delay: i * 0.1,
                      ease: "easeInOut"
                    }}
                  />
                ))}
              </svg>
            ) : (
              <motion.div 
                key="idle-line"
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: "80%", opacity: 0.2 }}
                className="h-[2px] bg-gradient-to-r from-transparent via-blue-500 to-transparent shadow-[0_0_15px_rgba(59,130,246,0.5)]"
              />
            )}
          </AnimatePresence>
        </div>
        
        {/* Glass Reflection */}
        <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/5 to-white/10 pointer-events-none" />
        <div className="absolute top-4 left-1/4 w-1/2 h-1/4 bg-white/5 rounded-[100%] blur-xl pointer-events-none" />
      </div>
    </div>
  );
};

// --- Vault Page Component ---

function VaultPage() {
  const navigate = useNavigate();
  const { filename } = Object.fromEntries(
    new URLSearchParams(window.location.search)
  );
  
  const actualFilename = window.location.pathname.split('/vault/')[1];
  const decodedFilename = actualFilename ? decodeURIComponent(actualFilename) : '';
  
  const [fileContent, setFileContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!decodedFilename) return;

    const fetchFile = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://localhost:8000/vault/files/${encodeURIComponent(decodedFilename)}`);
        if (!response.ok) throw new Error('Failed to load file');
        const data = await response.json();
        setFileContent(data.content);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchFile();
  }, [decodedFilename]);

  return (
    <div className="h-screen w-screen bg-[#070707] text-gray-200 flex flex-col p-6 font-sans overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 border-b border-[#262626] pb-4">
        <div className="flex items-center gap-4">
          <motion.button
            whileHover={{ scale: 1.1, x: -4 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/')}
            className="p-2 hover:bg-white/5 rounded-lg transition-all text-blue-400"
          >
            <ChevronLeft size={20} />
          </motion.button>
          <div>
            <h1 className="text-[11px] font-bold text-gray-400 tracking-[0.2em] uppercase">VAULT VIEWER</h1>
            <p className="text-[9px] text-gray-600 mt-1 font-mono">{decodedFilename}</p>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 bg-[#141414] border border-[#262626] rounded-xl overflow-hidden flex flex-col shadow-2xl">
        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin text-blue-400 text-3xl mb-4">⟳</div>
              <p className="text-[10px] uppercase tracking-widest text-gray-500">Loading file...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <p className="text-[10px] uppercase tracking-widest text-red-400 mb-2">✗ Error Loading File</p>
              <p className="text-[9px] text-gray-500">{error}</p>
            </div>
          </div>
        ) : (
          <div className="flex-1 bg-[#0d0d0d]/50 p-6 overflow-y-auto">
            <div className="prose prose-invert max-w-none">
              <div className="space-y-4">
                {fileContent.split('\n').map((line, i) => (
                  <div key={i} className="space-y-1">
                    {line.startsWith('# ') ? (
                      <h1 className="text-lg font-bold text-blue-400">{line.substring(2)}</h1>
                    ) : line.startsWith('## ') ? (
                      <h2 className="text-base font-bold text-blue-300 mt-4">{line.substring(3)}</h2>
                    ) : line.startsWith('### ') ? (
                      <h3 className="text-sm font-bold text-blue-200 mt-2">{line.substring(4)}</h3>
                    ) : line.startsWith('- ') || line.startsWith('* ') ? (
                      <div className="text-xs text-gray-300 pl-4">
                        • {line.substring(2)}
                      </div>
                    ) : line.startsWith('---') ? (
                      <div className="border-t border-[#262626] my-4" />
                    ) : line.trim() ? (
                      <p className="text-xs leading-relaxed text-gray-400 font-mono">{line}</p>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainApp />} />
        <Route path="/vault/:filename" element={<VaultPage />} />
      </Routes>
    </Router>
  );
}

function MainApp() {
  const navigate = useNavigate();
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<{text: string, isUser: boolean}[]>([]);
  const [logs, setLogs] = useState<{text: string, type: 'info' | 'tool' | 'error'}[]>([]);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [lastToolName, setLastToolName] = useState<string>('');
  const [transcript, setTranscript] = useState("");
  const [currentResponse, setCurrentResponse] = useState("");
  const [vaultFiles, setVaultFiles] = useState<{ filename: string; modified: number; size: number }[]>([]);
  const [vaultLoading, setVaultLoading] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);       // playback context (24 kHz)
  const captureContextRef = useRef<AudioContext | null>(null);     // capture context  (16 kHz)
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const transcriptRef = useRef<string>("");
  const userTranscriptRef = useRef<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch vault files
  const fetchVaultFiles = useCallback(async () => {
    setVaultLoading(true);
    try {
      const response = await fetch('http://localhost:8000/vault/files');
      if (!response.ok) throw new Error('Failed to fetch vault files');
      const data = await response.json();
      setVaultFiles(data.files || []);
      setLogs(prev => [...prev, { text: `✓ Loaded ${data.count || 0} vault files`, type: 'info' }]);
    } catch (err) {
      console.error('Error fetching vault files:', err);
      setLogs(prev => [...prev, { text: `✗ Failed to load vault files`, type: 'error' }]);
    } finally {
      setVaultLoading(false);
    }
  }, []);

  // Handle vault file click - navigate to vault page with filename
  const handleVaultFileClick = (filename: string) => {
    navigate(`/vault/${encodeURIComponent(filename)}`);
  };

  // Initial fetch of vault files
  useEffect(() => {
    fetchVaultFiles();
  }, [fetchVaultFiles]);

  // Refresh vault files every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchVaultFiles();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchVaultFiles]);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // FIX 2 — initAudio is async; resumes the playback context so audio plays back
  const initAudio = async () => {
    if (!audioContextRef.current) {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      await ctx.resume();
      audioContextRef.current = ctx;
    } else if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  };

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/voice`;
    let reconnectAttempt = 0;
    const maxReconnectAttempts = 5;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (socketRef.current && 
          (socketRef.current.readyState === WebSocket.OPEN || 
           socketRef.current.readyState === WebSocket.CONNECTING)) {
        console.log('[WS] Already connected, skipping duplicate connection');
        return;
      }
      
      try {
        console.log(`[WS] Connecting to ${wsUrl} (attempt ${reconnectAttempt + 1}/${maxReconnectAttempts})`);
        setLogs(prev => [...prev, { text: `⟳ Connecting to backend (attempt ${reconnectAttempt + 1}/${maxReconnectAttempts})...`, type: 'info' }]);

        const ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
          console.log('[WS] Connected successfully');
          reconnectAttempt = 0;
          setLogs(prev => [...prev, { text: "✓ Nova Sonic core connected", type: 'info' }]);
        };

        ws.onmessage = async (event) => {
          if (event.data instanceof ArrayBuffer) {
            playAudioChunk(event.data);
          } else {
            try {
              const data = JSON.parse(event.data);
              handleMetadata(data);
            } catch (e) {
              console.error('[WS] JSON parse error', e);
            }
          }
        };

        ws.onclose = () => {
          console.log(`[WS] Connection closed (readyState: ${ws.readyState})`);
          if (reconnectAttempt < maxReconnectAttempts) {
            reconnectAttempt++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempt - 1), 10000);
            console.log(`[WS] Scheduling reconnect in ${delay}ms`);
            setLogs(prev => [...prev, { text: `⟳ Reconnecting... (attempt ${reconnectAttempt}/${maxReconnectAttempts})`, type: 'error' }]);
            reconnectTimeout = setTimeout(connect, delay);
          } else {
            console.error('[WS] Max reconnect attempts exceeded');
            setLogs(prev => [...prev, { text: "✗ Cannot connect to backend. Is it running on port 8000?", type: 'error' }]);
          }
        };

        ws.onerror = (event) => {
          console.error('[WS] Connection error:', event);
          setLogs(prev => [...prev, {
            text: `✗ Backend connection failed. Ensure backend runs on port 8000.`,
            type: 'error'
          }]);
        };

        socketRef.current = ws;
      } catch (err) {
        console.error('[WS] Failed to create WebSocket:', err);
        setLogs(prev => [...prev, {
          text: `✗ WebSocket error: ${String(err)}`,
          type: 'error'
        }]);
      }
    };

    connect();
    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  // Debug: log messages array whenever it changes
  useEffect(() => {
    console.log('[STATE] Messages updated:', messages.length, 'items');
    if (messages.length > 0) {
      console.log('[MESSAGES]', JSON.stringify(messages, null, 2));
    }
  }, [messages]);

  const handleMetadata = useCallback((data: any) => {
    console.log('[METADATA EVENT]', data.type, data);

    switch (data.type) {
      case 'user_transcript':
        // Accumulate user speech and show it live in the centre display
        console.log('  -> User speech detected:', data.text);
        userTranscriptRef.current += data.text;
        setTranscript(userTranscriptRef.current);
        setLogs(prev => [...prev, { text: `You said: ${data.text}`, type: 'info' }]);
        break;

      case 'transcript':
        // Streaming assistant text chunks - accumulate and show live in centre
        console.log('  -> Model response chunk:', data.text);
        transcriptRef.current += data.text;
        setTranscript(transcriptRef.current);
        break;

      case 'response_complete': {
        // Commit the full turn to the chat panel - user bubble first, then assistant
        const userText = userTranscriptRef.current.trim();
        const assistantText = transcriptRef.current.trim();
        console.log('  -> Turn complete | user:', userText.slice(0, 50), '| assistant:', assistantText.slice(0, 50));
        setMessages(prev => {
          const next = [...prev];
          if (userText) next.push({ text: userText, isUser: true });
          if (assistantText) next.push({ text: assistantText, isUser: false });
          return next;
        });
        userTranscriptRef.current = "";
        transcriptRef.current = "";
        setTranscript("");
        setCurrentResponse("");
        setLogs(prev => [...prev, { text: "<- Response complete", type: 'info' }]);
        break;
      }

      // Legacy 'response' event fallback (backwards compat if backend sends it)
      case 'response': {
        const legacyAssistant = transcriptRef.current.trim() || (data.text || '').trim();
        const legacyUser = userTranscriptRef.current.trim();
        if (legacyUser || legacyAssistant) {
          setMessages(prev => {
            const next = [...prev];
            if (legacyUser) next.push({ text: legacyUser, isUser: true });
            if (legacyAssistant) next.push({ text: legacyAssistant, isUser: false });
            return next;
          });
        }
        userTranscriptRef.current = "";
        transcriptRef.current = "";
        setTranscript("");
        setCurrentResponse("");
        break;
      }

      case 'tool_call':
        console.log('  -> Tool invoked:', data.tool_name);
        setLogs(prev => [...prev, { text: `Invoking ${data.tool_name}`, type: 'tool' }]);
        break;

      case 'tool_result':
        console.log('  -> Tool result:', data.tool_name);
        setLastToolName(data.tool_name);
        setQueryResult(data.result);
        setLogs(prev => [...prev, { text: `${data.tool_name} returned data`, type: 'tool' }]);
        break;

      default:
        console.log('Unhandled metadata type:', data.type);
    }
  }, []);

  // nextPlayTimeRef tracks the end of the last scheduled audio chunk so each
  // new chunk is scheduled back-to-back, giving gapless (no-click) playback.
  const nextPlayTimeRef = useRef<number>(0);

  const playAudioChunk = useCallback((buffer: ArrayBuffer) => {
    if (!audioContextRef.current) {
      console.warn('[AUDIO] playAudioChunk called but audioContextRef is null');
      return;
    }
    if (buffer.byteLength === 0) return;

    try {
      const ctx = audioContextRef.current;
      const pcm16 = new Int16Array(buffer);
      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768;
      }

      const audioBuffer = ctx.createBuffer(1, float32.length, 24000);
      audioBuffer.getChannelData(0).set(float32);

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);

      // Schedule this chunk to start exactly when the previous one ends.
      // This eliminates the clicks and gaps that occur when each chunk is
      // played independently with source.start() (no time argument).
      const startAt = Math.max(ctx.currentTime, nextPlayTimeRef.current);
      source.start(startAt);
      nextPlayTimeRef.current = startAt + audioBuffer.duration;
    } catch (err) {
      console.error('[AUDIO] Error playing audio chunk:', err);
    }
  }, []);

  const startRecording = async () => {
    // FIX 2 — await the async initAudio so playback context is live before we start
    await initAudio();
    transcriptRef.current = "";
    userTranscriptRef.current = "";
    setTranscript("Listening...");

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        }
      });
      mediaStreamRef.current = stream;

      // FIX 1 & 3 — store in ref and call resume() so browser unblocks the audio graph
      captureContextRef.current = new AudioContext({ sampleRate: 16000 });
      await captureContextRef.current.resume();
      const source = captureContextRef.current.createMediaStreamSource(stream);

      // Load and initialize Web Audio Worklet
      try {
        console.log('[AUDIO] Loading audioWorklet from /audio-processor.js');
        setLogs(prev => [...prev, { text: "⟳ Loading audio processor...", type: 'info' }]);
        await captureContextRef.current.audioWorklet.addModule('/audio-processor.js');
        console.log('[AUDIO] ✓ AudioWorklet loaded successfully');
        setLogs(prev => [...prev, { text: "✓ Audio processor loaded", type: 'info' }]);
      } catch (workletErr) {
        const workletMsg = workletErr instanceof Error ? workletErr.message : String(workletErr);
        console.error('[AUDIO] Failed to load audioWorklet:', workletErr);
        setLogs(prev => [...prev, { text: `✗ AudioWorklet load failed: ${workletMsg}`, type: 'error' }]);
        throw new Error(`AudioWorklet initialization failed: ${workletMsg}`);
      }

      // FIX 3 — use captureContextRef.current everywhere (was split between local var and ref)
      const workletNode = new AudioWorkletNode(captureContextRef.current, 'audio-processor');

      let frameCount = 0;

      workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audio_chunk') {
          if (frameCount === 0) {
            console.log('[AUDIO] ✓ First audio chunk received from worklet!');
            setLogs(prev => [...prev, { text: "✓ Audio frames being captured", type: 'info' }]);
          }
          
          if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
            if (frameCount === 0) {
              console.warn('[AUDIO] WebSocket not ready when first chunk arrived');
              setLogs(prev => [...prev, { text: "✗ WebSocket not ready for audio", type: 'error' }]);
            }
            return;
          }
          try {
            socketRef.current.send(event.data.data);
            frameCount++;
            if (frameCount === 1) {
              console.log('[AUDIO] ✓ First audio chunk SENT to backend');
              setLogs(prev => [...prev, { text: "✓ Streaming audio to backend", type: 'info' }]);
            }
            if (frameCount % 100 === 0) {
              console.log(`[AUDIO] ${frameCount} frames sent`);
              setLogs(prev => {
                const lastLog = prev[prev.length - 1];
                if (lastLog?.text.includes('frames')) {
                  return [...prev.slice(0, -1), { text: `→ Streaming: ${frameCount} frames sent`, type: 'info' }];
                }
                return [...prev, { text: `→ Streaming: ${frameCount} frames sent`, type: 'info' }];
              });
            }
          } catch (err) {
            console.error('Error sending audio chunk:', err);
            setLogs(prev => [...prev, { text: `✗ Send error: ${String(err).slice(0, 50)}`, type: 'error' }]);
          }
        }
      };

      // FIX 3 — was captureContext.destination (undefined local var), now uses ref
      source.connect(workletNode);
      workletNode.connect(captureContextRef.current.destination);
      processorRef.current = workletNode;

      // Signal backend to start audio input (must happen before audio chunks are sent)
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        try {
          console.log('[AUDIO] Sending startAudio control message to backend');
          socketRef.current.send(JSON.stringify({ type: 'startAudio' }));
          setLogs(prev => [...prev, { text: "📤 Signaling backend: starting audio input", type: 'info' }]);
        } catch (err) {
          console.error('[AUDIO] Failed to send startAudio message:', err);
        }
      }

      setIsRecording(true);
      setLogs(prev => [...prev, { text: "✓ Microphone active - listening for speech", type: 'info' }]);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error("Microphone access error:", err);

      if (errorMsg.includes('Permission denied') || errorMsg.includes('NotAllowedError')) {
        setLogs(prev => [...prev, {
          text: "✗ Microphone permission denied. Allow access in browser settings.",
          type: 'error'
        }]);
      } else if (errorMsg.includes('not found') || errorMsg.includes('NotFoundError')) {
        setLogs(prev => [...prev, {
          text: "✗ No microphone found. Check device connection.",
          type: 'error'
        }]);
      } else {
        setLogs(prev => [...prev, {
          text: `✗ Mic error: ${errorMsg}`,
          type: 'error'
        }]);
      }
      setIsRecording(false);
    }
  };

  // FIX 3 — async so we can await captureContextRef.current.close()
  const stopRecording = async () => {
    try {
      if (processorRef.current) {
        processorRef.current.disconnect();
        if ('port' in processorRef.current) {
          (processorRef.current as any).port.close();
        }
        processorRef.current = null;
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
      }

      // FIX 3 — close and release the capture context to free mic hardware
      if (captureContextRef.current) {
        await captureContextRef.current.close();
        captureContextRef.current = null;
      }

      // Signal backend that audio input has ended
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        try {
          socketRef.current.send(JSON.stringify({ type: "endAudio" }));
          setLogs(prev => [...prev, { text: "⏹ Recording stopped - waiting for transcription...", type: 'info' }]);
        } catch (err) {
          console.error('Error sending endAudio signal:', err);
        }
      }

      setIsRecording(false);
      setTranscript("Processing...");
    } catch (err) {
      console.error('Error stopping recording:', err);
      setLogs(prev => [...prev, { text: "✗ Error stopping recording", type: 'error' }]);
    }
  };

  return (
    <div className="h-screen w-screen bg-[#070707] text-gray-200 flex p-6 gap-6 font-sans overflow-hidden">
      {/* Left Panel: Chat Session */}
      <div className="w-80 flex flex-col gap-6">
        <Panel title="Chat Session" icon={MessageSquare} className="flex-1">
          <div className="space-y-4 flex flex-col h-full">
            <div className="flex-1 overflow-y-auto flex flex-col gap-4">
              {messages.map((m, i) => (
                <MessageBubble key={i} text={m.text} isUser={m.isUser} />
              ))}
              {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center opacity-10 text-center">
                  <MessageSquare size={40} className="mb-4" />
                  <p className="text-xs uppercase tracking-widest font-bold">Encrypted Link Idle</p>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </Panel>
      </div>

      {/* Center Panel: Main Interface */}
      <div className="flex-1 flex flex-col gap-6">
        <div className="flex-1 bg-[#0a0a0a] border border-[#1a1a1a] rounded-3xl flex flex-col items-center justify-center relative shadow-[inset_0_0_100px_rgba(0,0,0,0.8)]">
          {/* Subtle Scanline Effect */}
          <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-20" />

          <VoiceVisualizer isActive={isRecording} />

          <div className="mt-12 text-center px-12 max-w-xl">
            <AnimatePresence mode="wait">
              {transcript && transcript !== "Listening..." ? (
                <motion.div
                  key="transcript"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className={cn(
                    "px-6 py-3 rounded-full backdrop-blur-md border",
                    userTranscriptRef.current && !transcriptRef.current
                      ? "bg-blue-500/10 border-blue-500/20"
                      : "bg-purple-500/10 border-purple-500/20"
                  )}
                >
                  <p className={cn(
                    "font-mono text-sm uppercase tracking-wider",
                    userTranscriptRef.current && !transcriptRef.current
                      ? "text-blue-400"
                      : "text-purple-400"
                  )}>
                    {transcript}
                  </p>
                </motion.div>
              ) : transcript === "Listening..." ? (
                <motion.div
                  key="listening"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-blue-500/10 border border-blue-500/20 px-6 py-3 rounded-full backdrop-blur-md"
                >
                  <p className="text-blue-400 font-mono text-sm uppercase tracking-wider animate-pulse">
                    Listening for your question...
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.3 }}
                  className="flex flex-col items-center gap-2"
                >
                  <p className="text-[10px] uppercase tracking-[0.3em] font-black text-gray-400">System Ready</p>
                  <div className="flex gap-1">
                    {[...Array(3)].map((_, i) => (
                      <div key={i} className="w-1 h-1 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.2}s` }} />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <motion.button
            whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(59, 130, 246, 0.4)" }}
            whileTap={{ scale: 0.95 }}
            // FIX 3 — wrap stopRecording in arrow fn so the async promise is handled correctly
            onClick={isRecording ? () => { stopRecording(); } : startRecording}
            className={cn(
              "mt-16 w-24 h-24 rounded-full flex items-center justify-center transition-all duration-500 border-2",
              isRecording
                ? "bg-red-500/10 border-red-500/50 text-red-500 shadow-[0_0_40px_rgba(239,68,68,0.2)]"
                : "bg-blue-600 border-blue-400 text-white shadow-[0_0_40px_rgba(37,99,235,0.3)]"
            )}
          >
            {isRecording ? <MicOff size={36} /> : <Mic size={36} />}
          </motion.button>
        </div>

        {/* Input fallback */}
        <div className="h-16 bg-[#111] border border-[#1a1a1a] rounded-2xl flex items-center px-6 gap-4 shadow-xl group">
          <Terminal size={18} className="text-blue-500/50 group-focus-within:text-blue-400 transition-colors" />
          <input
            type="text"
            placeholder="AWAITING INPUT COMMAND..."
            className="bg-transparent flex-1 border-none focus:outline-none text-[11px] font-mono tracking-widest text-gray-400 placeholder:text-gray-700"
          />
          <button className="p-2 hover:bg-white/5 rounded-xl transition-all text-blue-500 hover:text-blue-400">
            <Send size={18} />
          </button>
        </div>
      </div>

      {/* Right Panels */}
      <div className="w-80 flex flex-col gap-6">
        <Panel title="Vault Files" icon={Database} className="h-1/2">
          <div className="space-y-2 flex flex-col h-full">
            <div className="flex-1 overflow-y-auto flex flex-col gap-2">
              {vaultLoading ? (
                <div className="flex items-center justify-center h-full opacity-50">
                  <div className="animate-spin">⟳</div>
                  <p className="text-[10px] ml-2">Loading files...</p>
                </div>
              ) : vaultFiles.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center opacity-10 text-center">
                  <File size={32} className="mb-3" />
                  <p className="text-[10px] uppercase tracking-widest font-bold">No Files</p>
                </div>
              ) : (
                vaultFiles.map((file, i) => (
                  <VaultFileItem
                    key={i}
                    filename={file.filename}
                    modified={file.modified}
                    onClick={() => handleVaultFileClick(file.filename)}
                  />
                ))
              )}
            </div>
          </div>
        </Panel>

        <Panel title="Query Stream" icon={Activity} className="h-1/2">
          {queryResult ? (
            <FormattedResult data={queryResult} toolName={lastToolName} />
          ) : (
            <div className="h-full flex flex-col items-center justify-center opacity-10 text-center">
              <Activity size={40} className="mb-3 animate-pulse" />
              <p className="text-[10px] uppercase tracking-widest font-bold">No Data Flow</p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}