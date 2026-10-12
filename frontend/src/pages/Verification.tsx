import React, { useState, useRef, useEffect } from 'react';
import { Camera, StopCircle, CheckCircle, XCircle } from 'lucide-react';
import { useUsers, useCreateSession, useSubmitFrame } from '../api/hooks';
import type { FrameResult, OutcomeType } from '../types';

const outcomeConfig: Record<OutcomeType, { label: string; color: string; bg: string }> = {
  VERIFIED: { label: 'VERIFIED', color: 'text-green-400', bg: 'bg-green-900/30 border-green-700' },
  REJECTED: { label: 'REJECTED', color: 'text-red-400', bg: 'bg-red-900/30 border-red-700' },
  NO_FACE: { label: 'NO FACE DETECTED', color: 'text-yellow-400', bg: 'bg-yellow-900/30 border-yellow-700' },
  MULTIPLE_FACES: { label: 'MULTIPLE FACES', color: 'text-orange-400', bg: 'bg-orange-900/30 border-orange-700' },
  SPOOF_DETECTED: { label: 'SPOOF DETECTED', color: 'text-red-500', bg: 'bg-red-950/50 border-red-600' },
  ERROR: { label: 'ERROR', color: 'text-slate-400', bg: 'bg-slate-800 border-slate-600' },
  TIMEOUT: { label: 'TIMEOUT', color: 'text-slate-400', bg: 'bg-slate-800 border-slate-600' },
  LOCKOUT: { label: 'LOCKED OUT', color: 'text-red-500', bg: 'bg-red-950/50 border-red-600' },
};

const Verification: React.FC = () => {
  const [selectedUserId, setSelectedUserId] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [lastResult, setLastResult] = useState<FrameResult | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [capturing, setCapturing] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { data: usersData } = useUsers();
  const createSession = useCreateSession();
  const submitFrame = useSubmitFrame();

  const enrolledUsers = usersData?.users?.filter((u: any) => u.has_enrolment) || [];

  useEffect(() => {
    return () => {
      stopVerification();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (e) {
      console.error('Camera error:', e);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureFrame = (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!videoRef.current) { resolve(null); return; }
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) { resolve(null); return; }
      ctx.drawImage(videoRef.current, 0, 0);
      canvas.toBlob(resolve, 'image/jpeg', 0.85);
    });
  };

  const startVerification = async () => {
    if (!selectedUserId) return;

    await startCamera();

    try {
      const session = await createSession.mutateAsync({ target_user_id: selectedUserId });
      setIsRunning(true);
      setLastResult(null);

      const updateCountdown = () => {
        const exp = new Date(session.expires_at);
        const remaining = Math.max(0, Math.floor((exp.getTime() - Date.now()) / 1000));
        setCountdown(remaining);
        if (remaining <= 0) stopVerification();
      };
      updateCountdown();
      countdownRef.current = setInterval(updateCountdown, 1000);

      intervalRef.current = setInterval(async () => {
        if (capturing) return;
        setCapturing(true);
        try {
          const blob = await captureFrame();
          if (blob) {
            const result = await submitFrame.mutateAsync({ sessionId: session.session_id, file: blob });
            setLastResult(result);
            if (result.action !== 'retry') {
              setTimeout(() => stopVerification(), 2000);
            }
          }
        } catch (e) {
          console.error('Frame submission error:', e);
        } finally {
          setCapturing(false);
        }
      }, 1500);

    } catch (e: any) {
      console.error('Session creation error:', e);
      stopCamera();
      setIsRunning(false);
    }
  };

  const stopVerification = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (countdownRef.current) { clearInterval(countdownRef.current); countdownRef.current = null; }
    stopCamera();
    setIsRunning(false);
    setCountdown(0);
  };

  const result = lastResult ? outcomeConfig[lastResult.outcome] : null;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Live Verification</h2>

      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="relative bg-slate-900 rounded-xl border border-slate-700 overflow-hidden aspect-video">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover"
            />
            {!isRunning && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Camera size={48} className="text-slate-600" />
              </div>
            )}
            {isRunning && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="border-2 border-blue-400 border-dashed rounded-full"
                  style={{ width: '60%', height: '80%' }} />
              </div>
            )}
            {isRunning && countdown > 0 && (
              <div className="absolute top-3 right-3 bg-slate-900/80 text-white text-sm px-2 py-1 rounded">
                {countdown}s
              </div>
            )}
          </div>

          <div className="flex gap-3">
            {!isRunning ? (
              <button
                onClick={startVerification}
                disabled={!selectedUserId || createSession.isPending}
                className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg py-2 text-sm font-medium"
              >
                <Camera size={16} />
                Start Verification
              </button>
            ) : (
              <button
                onClick={stopVerification}
                className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 text-white rounded-lg py-2 text-sm font-medium"
              >
                <StopCircle size={16} />
                Stop
              </button>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 space-y-3">
            <label className="block text-sm text-slate-300">Select User to Verify</label>
            <select
              value={selectedUserId}
              onChange={e => setSelectedUserId(e.target.value)}
              disabled={isRunning}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm"
            >
              <option value="">-- Select enrolled user --</option>
              {enrolledUsers.map((u: any) => (
                <option key={u.id} value={u.id}>{u.full_name} ({u.email})</option>
              ))}
            </select>
            {enrolledUsers.length === 0 && (
              <p className="text-yellow-400 text-xs">No enrolled users found. Enrol users first.</p>
            )}
          </div>

          {lastResult && result && (
            <div className={`rounded-xl border p-4 ${result.bg}`}>
              {lastResult.outcome === 'VERIFIED' ? (
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="text-green-400" size={24} />
                  <span className="text-green-400 font-bold text-lg">AUTHENTICATED</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 mb-2">
                  <XCircle className={result.color} size={20} />
                  <span className={`font-bold ${result.color}`}>{result.label}</span>
                </div>
              )}
              <p className="text-sm text-slate-300 mb-3">{lastResult.explanation}</p>
              <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
                {lastResult.distance !== undefined && (
                  <div>Distance: <span className="text-white">{lastResult.distance.toFixed(3)}</span></div>
                )}
                {lastResult.threshold !== undefined && (
                  <div>Threshold: <span className="text-white">{lastResult.threshold.toFixed(3)}</span></div>
                )}
                <div>Latency: <span className="text-white">{lastResult.inference_ms.toFixed(0)}ms</span></div>
                <div>Attempts left: <span className="text-white">{lastResult.attempts_remaining}</span></div>
              </div>
            </div>
          )}

          {isRunning && !lastResult && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span className="text-slate-300 text-sm">Scanning... Look at the camera</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Verification;
