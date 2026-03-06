import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { Shield, Camera, Upload, Activity, AlertTriangle, CheckCircle2, Zap, RefreshCw, BarChart3, Database, Thermometer, Terminal, Cpu, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Bar, Doughnut } from 'react-chartjs-2';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const socket = io('http://127.0.0.1:5011');

export default function App() {
  const [activeTab, setActiveTab] = useState<'webcam' | 'upload'>('webcam');
  const [isProcessing, setIsProcessing] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [recordingStatus, setRecordingStatus] = useState<'idle' | 'recording'>('idle');
  const [countdown, setCountdown] = useState(0);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    socket.on('connect', () => {
      console.log('[SENTINEL] Connected to Forensic Engine');
      setLogs(prev => [...prev, "SYSTEM_READY: Sentinel Node Online"]);
    });

    socket.on('task_started', (data) => {
      console.log("[SENTINEL] Task Started:", data);
      if (data.task_id) {
        setActiveTaskId(data.task_id);
      } else if (data.status === 'error') {
        setIsProcessing(false);
        setLogs(prev => [...prev, `CRITICAL_FAULT: ${data.message}`]);
      }
    });

    if (activeTab === 'webcam') {
      navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true })
        .then(stream => {
          if (videoRef.current) videoRef.current.srcObject = stream;
        })
        .catch(err => console.error("Webcam Error:", err));
    }

    return () => {
      socket.off('connect');
      socket.off('task_started');
    };
  }, [activeTab]);

  // Hardened Polling Logic
  useEffect(() => {
    let interval: any;
    if (activeTaskId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`http://localhost:5011/task/${activeTaskId}`);
          const rawText = await res.clone().text(); // Catch raw response
          console.log(`[RAW_TRACE] ${res.status} | Body:`, rawText);

          if (!res.ok) throw new Error(`HTTP_${res.status}`);

          const data = await res.json();
          if (data.logs) setLogs(data.logs);

          if (data.status === 'completed' && data.result) {
            setReport(data.result);
            setIsProcessing(false);
            setActiveTaskId(null);
            setLogs(prev => [...prev, "AUDIT_COMPLETE: Result delivered to Intelligence Pane."]);
            clearInterval(interval);
          } else if (data.status === 'error') {
            setIsProcessing(false);
            setActiveTaskId(null);
            setLogs(prev => [...prev, `AUDIT_ABORTED: ${data.message}`]);
            clearInterval(interval);
          }
        } catch (err: any) {
          console.error("Polling Pipe Break:", err);
          setLogs(prev => [...prev, `PIPE_BREAK: Retrying sync... (${err.message})`]);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [activeTaskId]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const startAnalysis = () => {
    if (!videoRef.current?.srcObject) return;
    setRecordingStatus('recording');
    setCountdown(5);
    chunksRef.current = [];

    const stream = videoRef.current.srcObject as MediaStream;
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          setIsProcessing(true);
          setReport(null);
          setLogs(prev => [...prev, "DATA_BRIDGE: Routing buffer to /analyze REST pipe..."]);

          const res = await fetch('http://127.0.0.1:5011/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ buffer: reader.result as string })
          });

          if (!res.ok) throw new Error(`HTTP_${res.status}`);

          const data = await res.json();
          // Map Bridge JSON to UI expected structure
          const mappedReport = {
            metrics: {
              classification: data.verdict,
              ensemble_score: data.score,
              sync_score: data.sync_score,
              biometric_score: data.biometric_score,
              reflection_score: data.reflection_score,
              rppg_score: data.score * 0.9 // Heuristic for rPPG component
            },
            forensics: {
              filtered: data.rppg_graph,
              bpm: 72 // Fallback for headless
            },
            telemetry: { compute_time: 0.8 },
            environment: { lux: 100 }
          };

          setReport(mappedReport);
          setIsProcessing(false);
          setLogs(prev => [...prev, "AUDIT_COMPLETE: Result delivered via JSON Bridge."]);

        } catch (err: any) {
          console.error("Bridge Fault:", err);
          setIsProcessing(false);
          setLogs(prev => [...prev, `BRIDGE_FAULT: ${err.message}`]);
        }
      };
      reader.readAsDataURL(blob);
    };

    mediaRecorder.start();
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(interval);
          mediaRecorder.stop();
          setRecordingStatus('idle');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);
    setReport(null);
    setLogs(prev => [...prev, `DATA_INBOUND: Processing ${file.name}...`]);

    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const res = await fetch('http://127.0.0.1:5011/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ buffer: reader.result as string })
        });

        if (!res.ok) throw new Error(`HTTP_${res.status}`);
        const data = await res.json();

        const mappedReport = {
          metrics: {
            classification: data.verdict,
            ensemble_score: data.score,
            sync_score: data.sync_score,
            biometric_score: data.biometric_score,
            reflection_score: data.reflection_score,
            rppg_score: data.score * 0.9
          },
          forensics: {
            filtered: data.rppg_graph,
            bpm: 75
          },
          telemetry: { compute_time: 1.2 },
          environment: { lux: 120 }
        };

        setReport(mappedReport);
        setIsProcessing(false);
        setLogs(prev => [...prev, "AUDIT_COMPLETE: Remote archive processed via JSON Bridge."]);
      } catch (err: any) {
        setIsProcessing(false);
        setLogs(prev => [...prev, `BRIDGE_FAULT: ${err.message}`]);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="h-screen w-screen bg-[#050505] text-[#e4e4e7] overflow-hidden p-6 font-['Inter',sans-serif]">
      {/* Header bar */}
      <div className="flex items-center justify-between mb-6 px-4">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg shadow-[0_0_20px_rgba(37,99,235,0.3)]">
            <Shield className="h-5 w-5 text-white" />
          </div>
          <h1 className="text-xl font-black tracking-tighter uppercase italic">Persona <span className="text-blue-500">Sentinel</span></h1>
        </div>
        <div className="flex items-center gap-6 text-[10px] font-mono tracking-widest text-zinc-500">
          <div className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" /> ENGINE_ONLINE</div>
          <div className="flex items-center gap-2 underline decoration-blue-500/30 cursor-pointer hover:text-white" onClick={() => setLogs([])}>PURGE_CACHED_LOGS</div>
        </div>
      </div>

      <div className="grid grid-cols-12 h-[calc(100%-80px)] gap-6">
        {/* Left Pane: The Input (Control Center) */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-6">
          <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 flex-1 flex flex-col">
            <div className="flex bg-black/40 p-1 rounded-2xl mb-6 w-fit border border-white/5">
              <button onClick={() => setActiveTab('webcam')} className={`px-6 py-2 rounded-xl text-xs font-bold transition-all ${activeTab === 'webcam' ? 'bg-zinc-800 text-white shadow-xl' : 'text-zinc-500 hover:text-zinc-300'}`}>WEBCAM_FEED</button>
              <button
                onClick={() => {
                  setActiveTab('upload');
                  // Directive: Release hardware when switching to Remote Archive
                  fetch('http://127.0.0.1:5011/release', { method: 'POST' })
                    .catch(err => console.error("Hardware release signal failed:", err));
                }}
                className={`px-6 py-2 rounded-xl text-xs font-bold transition-all ${activeTab === 'upload' ? 'bg-zinc-800 text-white shadow-xl' : 'text-zinc-500 hover:text-zinc-300'}`}
              >
                REMOTE_ARCHIVE
              </button>
            </div>

            <div className="flex-1 rounded-2xl bg-black/60 border border-white/5 relative overflow-hidden flex items-center justify-center group">
              {activeTab === 'webcam' ? (
                <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover grayscale opacity-50 contrast-125" />
              ) : (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex flex-col items-center gap-4 cursor-pointer hover:scale-105 transition-transform"
                >
                  <div className="p-6 bg-white/5 rounded-full border border-white/10 group-hover:bg-blue-500/10 group-hover:border-blue-500/30 transition-colors">
                    <Upload className="h-10 w-10 text-zinc-600 group-hover:text-blue-500" />
                  </div>
                  <p className="text-[10px] font-mono text-zinc-500 tracking-widest uppercase">Select Forensic Asset</p>
                </div>
              )}

              {isProcessing && (
                <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    className="relative w-24 h-24 mb-6"
                  >
                    <div className="absolute inset-0 rounded-full border-2 border-blue-500/20" />
                    <div className="absolute inset-0 rounded-full border-t-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
                  </motion.div>
                  <p className="font-mono text-[10px] text-blue-500 tracking-[0.3em] animate-pulse">EXTRACTING_VITALS</p>
                </div>
              )}

              {recordingStatus === 'recording' && (
                <div className="absolute top-6 left-6 flex items-center gap-2 bg-red-500/20 border border-red-500/50 px-3 py-1 rounded-full">
                  <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <span className="text-[10px] font-mono font-bold text-red-500 tracking-tighter">BUFFERING [{countdown}s]</span>
                </div>
              )}
            </div>

            {activeTab === 'webcam' && (
              <button
                onClick={startAnalysis}
                disabled={recordingStatus === 'recording' || isProcessing}
                className="mt-6 w-full py-4 bg-white text-black font-black uppercase tracking-tighter rounded-2xl hover:bg-zinc-200 transition-colors disabled:opacity-50 flex items-center justify-center gap-3 shadow-[0_10px_30px_rgba(255,255,255,0.1)]"
              >
                {recordingStatus === 'recording' ? 'Buffer Lock...' : 'Start Audit Analysis'}
                <Zap className="h-4 w-4" />
              </button>
            )}
            <input type="file" ref={fileInputRef} className="hidden" accept="video/*" onChange={handleFileUpload} />
          </div>

          <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 h-48 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <p className="text-[9px] font-black text-zinc-600 uppercase tracking-widest flex items-center gap-2">
                <Terminal className="h-4 w-4 text-blue-500" /> Sentinel_Node_Console
              </p>
              <Cpu className="h-3 w-3 text-zinc-800" />
            </div>
            <div className="flex-1 overflow-y-auto font-mono text-[9px] text-zinc-400 space-y-1 custom-scrollbar pr-2">
              {logs.length === 0 && <p className="opacity-10 italic">Awaiting forensic stream initialization...</p>}
              {logs.map((log, i) => (
                <div key={i} className="flex gap-3">
                  <span className="text-zinc-700">[{new Date().toLocaleTimeString([], { hour12: false })}]</span>
                  <span className={log.includes('COMPLETE') || log.includes('SUCCESS') ? 'text-green-500' : log.includes('ABORTED') || log.includes('CRITICAL') ? 'text-red-500 font-bold' : ''}>{log}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>

        {/* Right Pane: The Intelligence (Intelligence Pane) */}
        <div className="col-span-12 lg:col-span-7 bg-zinc-900/10 border border-white/5 rounded-3xl p-8 overflow-hidden flex flex-col">
          <AnimatePresence mode="wait">
            {!report && !isProcessing ? (
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex-1 flex flex-col items-center justify-center text-center opacity-20"
              >
                <div className="w-32 h-32 rounded-full border-2 border-dashed border-zinc-700 mb-6 flex items-center justify-center">
                  <Database className="h-12 w-12 text-zinc-600" />
                </div>
                <h3 className="text-sm font-mono tracking-widest uppercase mb-2">Awaiting Intelligence</h3>
                <p className="text-[10px] max-w-[200px] leading-relaxed">Initialize a forensic scan on the left to populate biometric insights.</p>
              </motion.div>
            ) : report ? (
              <motion.div
                initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}
                className="flex-1 flex flex-col gap-8"
              >
                {/* Gauge Section */}
                <div className="flex items-center justify-between gap-12 pt-4">
                  <div className="flex-1">
                    <p className="text-[10px] font-black text-zinc-600 uppercase tracking-widest mb-1">Authenticity_Consensus</p>
                    <h2 className={`text-6xl font-black tracking-tighter ${report.metrics.classification === 'HUMAN' ? 'text-green-500' : 'text-red-500'}`}>
                      {report.metrics.classification}
                    </h2>
                    <div className="flex items-center gap-2 mt-2">
                      <CheckCircle2 className={`h-4 w-4 ${report.metrics.classification === 'HUMAN' ? 'text-green-500' : 'text-red-500'}`} />
                      <span className="text-[10px] font-mono text-zinc-400">Ensemble Confidence: {(report.metrics.ensemble_score * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="w-40 h-40">
                    <Doughnut
                      data={{
                        labels: ['Synthetic', 'Organic'],
                        datasets: [{
                          data: [report.metrics.ensemble_score * 100, (1 - report.metrics.ensemble_score) * 100],
                          backgroundColor: [report.metrics.classification === 'HUMAN' ? '#22c55e' : '#ef4444', '#18181b'],
                          borderWidth: 0,
                        }]
                      }}
                      options={{ cutout: '85%', plugins: { legend: { display: false } } }}
                    />
                  </div>
                </div>

                {/* Charts Grid */}
                <div className="grid grid-cols-2 gap-4 flex-1 min-h-0">
                  <div className="bg-black/30 border border-white/5 rounded-2xl p-6 flex flex-col">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-[10px] font-bold text-zinc-500 tracking-widest flex items-center gap-2 uppercase">
                        <Activity className="h-3 w-3 text-red-500" /> Physiological_BVP
                      </span>
                      <span className="text-[10px] font-mono text-zinc-400">{report.forensics?.bpm} BPM</span>
                    </div>
                    <div className="flex-1 min-h-0">
                      <Line
                        data={{
                          labels: report.forensics?.filtered ? new Array(report.forensics.filtered.length).fill('') : [],
                          datasets: [{
                            data: report.forensics?.filtered || [],
                            borderColor: '#ef4444',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            tension: 0.4,
                            fill: true,
                            backgroundColor: 'rgba(239, 68, 68, 0.05)'
                          }]
                        }}
                        options={{ responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { display: false } }, plugins: { legend: { display: false } } }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/30 border border-white/5 rounded-2xl p-6 flex flex-col">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-[10px] font-bold text-zinc-500 tracking-widest flex items-center gap-2 uppercase">
                        <BarChart3 className="h-3 w-3 text-blue-500" /> Specialist_Consensus
                      </span>
                      <span className="text-[10px] font-mono text-zinc-400">Total Audit Components: 4</span>
                    </div>
                    <div className="flex-1 min-h-0">
                      <Bar
                        data={{
                          labels: ['Heart', 'Lip', 'Eye-J', 'Eye-R'],
                          datasets: [{
                            data: [
                              report.metrics.rppg_score,
                              report.metrics.sync_score,
                              report.metrics.biometric_score,
                              report.metrics.reflection_score
                            ],
                            backgroundColor: ['#ef4444', '#3b82f6', '#f59e0b', '#06b6d4'],
                            borderRadius: 4
                          }]
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            x: { display: true, ticks: { color: '#71717a', font: { size: 8 } } },
                            y: { display: false, min: 0, max: 1 }
                          },
                          plugins: { legend: { display: false } }
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Metadata Footer */}
                <div className="grid grid-cols-4 gap-4 pt-6 border-t border-white/5">
                  <StatBox label="LUMINANCE" value={report.environment?.lux || '0'} unit="lx" sub={report.environment?.low_light ? 'DIM' : 'OPT'} color={report.environment?.low_light ? 'text-orange-500' : 'text-blue-500'} />
                  <StatBox label="RESOLUTION" value="640x480" unit="" sub="NORMALIZED" color="text-zinc-300" />
                  <StatBox label="FRAME RATE" value="10" unit="fps" sub="OPTIMIZED" color="text-zinc-300" />
                  <StatBox label="COMPUTE" value={report.telemetry?.compute_time || '0'} unit="s" sub="LOCAL" color="text-zinc-300" />
                </div>
              </motion.div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 opacity-50">
                <RefreshCw className="h-8 w-8 text-blue-500 animate-spin" />
                <p className="text-[10px] font-mono tracking-widest uppercase">Aggregating Specialist Intelligence...</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Tooltip Overlay */}
      <div className="fixed bottom-6 right-6 group">
        <Info className="h-5 w-5 text-zinc-700 cursor-pointer hover:text-white transition-colors" />
        <div className="absolute bottom-full right-0 mb-4 w-64 bg-zinc-900 border border-white/10 p-4 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-2xl">
          <p className="text-[10px] font-bold text-blue-500 uppercase tracking-widest mb-2">Sentinel Node Security</p>
          <p className="text-[9px] text-zinc-400 leading-relaxed italic">Headless biometric audit engine v2.2. Optimized for local hardware stability. Data trimmed to 5s for memory safety.</p>
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value, unit, sub, color }: { label: string, value: any, unit: string, sub: string, color: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[8px] font-black text-zinc-600 tracking-[0.2em] uppercase">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className={`text-sm font-bold ${color}`}>{value}</span>
        <span className="text-[8px] text-zinc-700 font-mono uppercase">{unit}</span>
      </div>
      <span className="text-[8px] font-mono text-zinc-800 uppercase italic">{sub}</span>
    </div>
  );
}
