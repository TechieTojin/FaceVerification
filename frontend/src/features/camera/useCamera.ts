import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraStatus = 'idle' | 'requesting' | 'active' | 'denied' | 'notfound' | 'inuse' | 'unsupported' | 'error' | 'ended';

export const CAMERA_MESSAGES: Record<Exclude<CameraStatus, 'idle' | 'requesting' | 'active'>, string> = {
  denied: 'Camera permission was denied. Allow camera access in your browser settings and try again.',
  notfound: 'No camera was found. Connect a camera and try again.',
  inuse: 'The camera is in use by another application, or could not be started.',
  unsupported: 'This browser does not support camera access (a secure https or localhost origin is required).',
  error: 'The camera could not be started.',
  ended: 'The camera stopped unexpectedly (it may have been disconnected).',
};

export function classifyCameraError(err: unknown): CameraStatus {
  const name = (err as { name?: string })?.name ?? '';
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') return 'denied';
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') return 'notfound';
  if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') return 'inuse';
  return 'error';
}

export interface CameraApi {
  status: CameraStatus;
  devices: MediaDeviceInfo[];
  deviceId: string | null;
  setDeviceId: (id: string) => void;
  start: () => Promise<boolean>;
  stop: () => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

/** Camera lifecycle with guaranteed track cleanup on stop, device switch and unmount. */
export function useCamera(): CameraApi {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceIdState] = useState<string | null>(null);
  const deviceRef = useRef<string | null>(null);
  const generation = useRef(0);

  const release = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => {
      t.onended = null;
      t.stop();
    });
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const refreshDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === 'videoinput'));
    } catch {
      setDevices([]);
    }
  }, []);

  const start = useCallback(async (): Promise<boolean> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('unsupported');
      return false;
    }
    const gen = ++generation.current;
    release();
    setStatus('requesting');
    try {
      const video: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' };
      if (deviceRef.current) video.deviceId = { exact: deviceRef.current };
      const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      if (gen !== generation.current) {          // superseded by stop()/another start()
        stream.getTracks().forEach((t) => t.stop());
        return false;
      }
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      if (track) {
        track.onended = () => {
          if (gen === generation.current) setStatus('ended');
        };
        const id = track.getSettings().deviceId;
        if (id && !deviceRef.current) {
          deviceRef.current = id;
          setDeviceIdState(id);
        }
      }
      const el = videoRef.current;
      if (el) {
        el.srcObject = stream;
        await el.play().catch(() => undefined);
      }
      setStatus('active');
      void refreshDevices();
      return true;
    } catch (err) {
      if (gen === generation.current) setStatus(classifyCameraError(err));
      return false;
    }
  }, [release, refreshDevices]);

  const stop = useCallback(() => {
    generation.current++;
    release();
    setStatus('idle');
  }, [release]);

  const setDeviceId = useCallback((id: string) => {
    deviceRef.current = id;
    setDeviceIdState(id);
    if (streamRef.current) void start();
  }, [start]);

  useEffect(() => {
    const md = navigator.mediaDevices;
    md?.addEventListener?.('devicechange', refreshDevices);
    return () => {
      md?.removeEventListener?.('devicechange', refreshDevices);
      generation.current++;
      release();
    };
  }, [refreshDevices, release]);

  return { status, devices, deviceId, setDeviceId, start, stop, videoRef };
}
