import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { Shield, Camera, Upload, Activity, AlertTriangle, CheckCircle2, Zap, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import { FaceMesh } from '@mediapipe/face_mesh';
import * as cam from '@mediapipe/camera_utils';
import { drawConnectors } from '@mediapipe/drawing_utils';
import { FACEMESH_TESSELATION } from '@mediapipe/face_mesh';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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
  Title,
  Tooltip,
  Legend,
  Filler
);

const socket = io('http://localhost:5001');

export default function App() {
  const [activeTab, setActiveTab] = useState<'webcam' | 'upload'>('webcam');
  const [isProcessing, setIsProcessing] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [pulseData, setPulseData] = useState<number[]>(new Array(50).fill(0));

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    socket.on('analysis_complete', (data) => {
      setReport(data);
      setIsProcessing(false);
    });

    socket.on('rppg_update', (data) => {
      setPulseData(prev => [...prev.slice(1), data.value]);
    });

    let camera: any = null;
    let faceMesh: any = null;

    if (activeTab === 'webcam') {
      faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults((results: any) => {
        if (!canvasRef.current || !videoRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // Draw the video frame
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvasRef.current.width, 0);
        ctx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.restore();

        if (results.multiFaceLandmarks) {
          for (const landmarks of results.multiFaceLandmarks) {
            // We draw on the flipped canvas
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-canvasRef.current.width, 0);
            drawConnectors(ctx, landmarks, FACEMESH_TESSELATION, { color: '#10b981', lineWidth: 0.5 });
            ctx.restore();
          }
        }
      });

      if (videoRef.current) {
        camera = new cam.Camera(videoRef.current, {
          onFrame: async () => {
            if (videoRef.current) await faceMesh.send({ image: videoRef.current });
          },
          width: 640,
          height: 480,
        });
        camera.start();
      }
    }

    return () => {
      socket.off('analysis_complete');
      socket.off('rppg_update');
      if (camera) camera.stop();
      if (faceMesh) faceMesh.close();
    };
  }, [activeTab]);

  const triggerUpload = () => {
    setIsProcessing(true);
    setReport(null);
    socket.emit('trigger_analysis', { path: "c:/Users/Srinath/OneDrive/Desktop/Testing/data_10_real.mp4" });
  };

  return (
    <div className="flex h-screen w-screen bg-background p-4 gap-4">
      {/* Sidebar Navigation */}
      <div className="w-20 cyber-panel flex flex-col items-center py-8 gap-8">
        <div className="relative group">
          <Shield className="text-accent h-8 w-8 mb-4 cursor-help" />
          <div className="absolute left-14 top-0 bg-panel border border-border px-3 py-1 rounded text-[10px] text-zinc-400 opacity-0 group-hover:opacity-100 whitespace-nowrap pointer-events-none transition-opacity">PERSONA_CORE_v2.3</div>
        </div>
        <button
          onClick={() => setActiveTab('webcam')}
          className={`p-3 rounded-lg transition-all ${activeTab === 'webcam' ? 'bg-accent/20 text-accent ring-1 ring-accent/30' : 'text-zinc-500 hover:text-white'}`}
        >
          <Camera h-6 w-6 />
        </button>
        <button
          onClick={() => setActiveTab('upload')}
          className={`p-3 rounded-lg transition-all ${activeTab === 'upload' ? 'bg-accent/20 text-accent ring-1 ring-accent/30' : 'text-zinc-500 hover:text-white'}`}
        >
          <Upload h-6 w-6 />
        </button>
        <button onClick={() => window.location.reload()} className="p-3 mt-auto text-zinc-600 hover:text-zinc-400 transition-colors">
          <RefreshCw h-5 w-5 />
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Input Module */}
        <div className="flex-1 cyber-panel relative group bg-black/40">
          <div className="absolute top-4 left-4 z-10 flex gap-2">
            <span className="px-3 py-1 bg-black/80 backdrop-blur-md border border-border rounded-full text-[10px] text-accent font-bold tracking-widest flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-accent animate-pulse" />
              {activeTab === 'webcam' ? 'FEED.BIOMETRIC_TRACKING:ACTIVE' : 'FEED.ARCHIVE_EXTRACTION:WAITING'}
            </span>
          </div>

          <div className="h-full w-full flex items-center justify-center relative">
            {activeTab === 'webcam' ? (
              <>
                <video ref={videoRef} className="hidden" />
                <canvas ref={canvasRef} className="h-full w-full object-contain opacity-70" width={640} height={480} />
                {/* Visual HUD Overlays */}
                <div className="absolute inset-0 pointer-events-none p-12 flex flex-col justify-between">
                  <div className="flex justify-between">
                    <div className="w-8 h-8 border-t-2 border-l-2 border-accent/40 rounded-tl-lg" />
                    <div className="w-8 h-8 border-t-2 border-r-2 border-accent/40 rounded-tr-lg" />
                  </div>
                  <div className="flex justify-between">
                    <div className="w-8 h-8 border-b-2 border-l-2 border-accent/40 rounded-bl-lg" />
                    <div className="w-8 h-8 border-b-2 border-r-2 border-accent/40 rounded-br-lg" />
                  </div>
                </div>
              </>
            ) : (
              <div onClick={triggerUpload} className="cursor-pointer group text-center border-2 border-dashed border-border p-12 rounded-2xl hover:border-accent/40 hover:bg-accent/5 transition-all">
                <Upload className="h-16 w-16 text-zinc-700 mx-auto mb-4 group-hover:text-accent group-hover:scale-110 transition-all" />
                <p className="text-zinc-400 font-bold mb-1">Drag & Drop Forensic Archive</p>
                <p className="text-zinc-600 text-[10px] uppercase tracking-widest">MP4 / MOV / AVI (MAX 50MB)</p>
              </div>
            )}
          </div>

          <AnimatePresence>
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-20 backdrop-blur-md"
              >
                <div className="relative mb-8">
                  <Activity className="h-16 w-16 text-accent animate-pulse" />
                  <Zap className="absolute top-0 right-0 h-4 w-4 text-white animate-bounce" />
                </div>
                <p className="text-white font-mono text-xl tracking-[0.3em] font-black glitch-text">EXTRACTING_VITALS</p>
                <div className="w-80 h-[1px] bg-zinc-800 rounded-full mt-8 relative overflow-hidden">
                  <motion.div
                    className="absolute inset-y-0 w-1/4 bg-accent shadow-[0_0_15px_rgba(59,130,246,0.6)]"
                    animate={{ left: ["-25%", "125%"] }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                  />
                </div>
                <p className="text-[10px] text-zinc-500 mt-6 uppercase tracking-[0.2em] font-mono">Parallel Specialist Consensus in Progress...</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Live Diagnostics */}
        <div className="h-56 flex gap-4">
          <div className="flex-[1.5] cyber-panel p-4 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest flex items-center gap-2">
                <Activity h-3 w-3 className="text-human" />
                Biological Pulse Wave (rPPG)
              </span>
              <div className="flex gap-2">
                <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-[8px] text-zinc-400 font-mono">BUFFER: 50pts</span>
                <span className="px-2 py-0.5 rounded-full bg-human/20 text-[8px] text-human font-bold font-mono">LIVE_STREAM</span>
              </div>
            </div>
            <div className="flex-1 min-h-0">
              <Line
                data={{
                  labels: new Array(50).fill(''),
                  datasets: [{
                    data: pulseData,
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true,
                    backgroundColor: (context) => {
                      const chart = context.chart;
                      const { ctx, chartArea } = chart;
                      if (!chartArea) return null;
                      const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
                      gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');
                      return gradient;
                    }
                  }]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: { display: false },
                    y: {
                      display: true,
                      grid: { color: 'rgba(255,255,255,0.03)' },
                      ticks: { display: false },
                      min: -1.5,
                      max: 1.5
                    }
                  },
                  plugins: { legend: { display: false } },
                  animation: { duration: 0 }
                }}
              />
            </div>
          </div>
          <div className="flex-1 cyber-panel p-4 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest flex items-center gap-2">
                <Zap h-3 w-3 className="text-accent" />
                Lip-Sync Phase Map
              </span>
              <span className="text-[8px] text-zinc-600 font-mono">SPEC: SYNC-v1.4</span>
            </div>
            <div className="flex-1 flex items-end gap-1.5 px-2">
              {[40, 75, 45, 95, 65, 30, 85, 55, 75, 50, 60, 40].map((h, i) => (
                <div key={i} className="flex-1 bg-accent/10 border-t border-accent/30 rounded-t-sm" style={{ height: `${h}%` }} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Forensic Report Sidebar */}
      <div className="w-96 flex flex-col gap-4">
        <div className="flex-1 cyber-panel p-6 flex flex-col overflow-y-auto">
          <div className="border-b border-border pb-4 mb-6 flex justify-between items-end">
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight uppercase italic">Forensic_Audit</h2>
              <p className="text-[10px] text-zinc-500 font-mono">ID: PERS-{Math.random().toString(36).substr(2, 6).toUpperCase()}</p>
            </div>
            {report && (
              <span className={`px-2 py-1 rounded text-[10px] font-bold ${report.metrics.classification === 'HUMAN' ? 'bg-human/20 text-human border border-human/30' : 'bg-deepfake/20 text-deepfake border border-deepfake/30'}`}>
                {report.metrics.classification}
              </span>
            )}
          </div>

          {!report ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 border border-border rounded-full flex items-center justify-center mb-6 opacity-30">
                <Shield className="h-8 w-8 text-zinc-500" />
              </div>
              <p className="text-zinc-500 text-sm font-bold uppercase tracking-widest">Awaiting Analysis</p>
              <p className="text-[10px] text-zinc-600 mt-2 leading-relaxed">System in standby mode. Provide video stream or biometric archive to initiate specialist consensus.</p>
            </div>
          ) : (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="flex-1 flex flex-col">
              {/* Hero Gauge */}
              <div className="relative h-56 w-56 mx-auto flex items-center justify-center mb-8">
                <svg className="w-full h-full -rotate-90">
                  <circle cx="112" cy="112" r="90" className="stroke-zinc-900 fill-none" strokeWidth="6" />
                  <motion.circle
                    cx="112" cy="112" r="90"
                    className={`fill-none ${report.metrics.ensemble_score > 0.5 ? 'stroke-deepfake' : 'stroke-human'}`}
                    strokeWidth="8"
                    strokeDasharray="565.48"
                    initial={{ strokeDashoffset: 565.48 }}
                    animate={{ strokeDashoffset: 565.48 - (565.48 * report.metrics.ensemble_score) }}
                    transition={{ duration: 2, ease: "circOut" }}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-5xl font-black text-white tracking-tighter">{(report.metrics.ensemble_score * 100).toFixed(0)}</span>
                  <span className="text-[8px] text-zinc-500 tracking-[0.4em] uppercase font-bold mt-1">Probability_Index</span>
                </div>
              </div>

              {/* Forensic Metrics */}
              <div className="space-y-4 mb-8">
                <div className="grid grid-cols-2 gap-3">
                  <MetricBox label="BPM" value={`${report.forensics.bpm}`} sub="Heart Rate" />
                  <MetricBox label="SNR" value={`${report.forensics.snr.toFixed(1)}dB`} sub="Signal/Noise" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <MetricBox label="Sync" value={`${report.forensics.sync_corr.toFixed(3)}`} sub="Behavioral" />
                  <MetricBox label="Entropy" value={`${report.forensics.entropy.toFixed(2)}`} sub="Information" />
                </div>
              </div>

              {/* Environmental Alerts */}
              <div className="space-y-3 mt-auto">
                <p className="text-[10px] text-zinc-600 font-bold uppercase tracking-widest px-1">Contextual_Safeguards</p>
                {report.environment.low_light || report.environment.shaky || report.environment.grainy ? (
                  <>
                    <EnvAlert
                      label="LUMINANCE_CRITICAL"
                      active={report.environment.low_light}
                      msg="Low light detected. Defaulting to Lip-Sync bias (80%)."
                    />
                    <EnvAlert
                      label="STABILITY_WARPED"
                      active={report.environment.shaky}
                      msg="Handheld shake detected. Relaxed SNR applied."
                    />
                  </>
                ) : (
                  <div className="p-4 bg-human/5 border border-human/10 rounded-lg flex items-center gap-3">
                    <CheckCircle2 className="h-4 w-4 text-human" />
                    <p className="text-[10px] text-human font-bold uppercase">Optimal Conditions Detected</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricBox({ label, value, sub }: { label: string, value: string, sub: string }) {
  return (
    <div className="p-3 bg-panel border border-border rounded-lg relative overflow-hidden group hover:border-accent/40 transition-colors">
      <div className="absolute top-0 right-0 p-1 opacity-10">
        <Activity className="h-4 w-4 text-white" />
      </div>
      <p className="text-[9px] text-zinc-600 uppercase font-bold tracking-widest mb-1">{label}</p>
      <p className="text-xl font-bold text-white tracking-tight">{value}</p>
      <p className="text-[8px] text-accent/60 font-medium uppercase mt-1">{sub}</p>
    </div>
  );
}

function EnvAlert({ label, active, msg }: { label: string, active: boolean, msg: string }) {
  if (!active) return null;
  return (
    <div className="flex gap-4 p-4 bg-deepfake/5 border border-deepfake/20 rounded-xl relative">
      <div className="absolute inset-y-0 left-0 w-1 bg-deepfake/40 rounded-l-xl" />
      <AlertTriangle className="h-5 w-5 text-deepfake flex-shrink-0" />
      <div>
        <p className="text-[10px] font-black text-deepfake leading-none uppercase tracking-widest">{label}</p>
        <p className="text-[9px] text-zinc-400 mt-2 leading-relaxed">{msg}</p>
      </div>
    </div>
  );
}
