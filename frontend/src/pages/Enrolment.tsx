import React, { useState, useRef } from 'react';
import { Camera, CheckCircle, Upload } from 'lucide-react';
import { useUsers, useEnrolmentStatus, useRecordConsent, useEnrolUser } from '../api/hooks';

const Enrolment: React.FC = () => {
  const [selectedUserId, setSelectedUserId] = useState('');
  const [step, setStep] = useState<'select' | 'consent' | 'capture' | 'done'>('select');
  const [consentGiven, setConsentGiven] = useState(false);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [feedback, setFeedback] = useState('');
  const [error, setError] = useState('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const { data: usersData } = useUsers();
  const { data: enrolmentStatus, refetch: refetchStatus } = useEnrolmentStatus(selectedUserId || null);
  const recordConsent = useRecordConsent();
  const enrolUser = useEnrolUser();

  const users = usersData?.users || [];

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (e) {
      setError('Camera access denied');
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  const handleConsentSubmit = async () => {
    setError('');
    try {
      await recordConsent.mutateAsync({
        userId: selectedUserId,
        consent_given: consentGiven,
        consent_text: 'User has given informed consent to biometric data collection.',
      });
      if (consentGiven) {
        setStep('capture');
        await startCamera();
      }
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Failed to record consent');
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(videoRef.current, 0, 0);
    canvas.toBlob(blob => {
      if (blob) {
        setCapturedBlob(blob);
        setFeedback('Photo captured! Click "Enrol" to complete enrolment.');
      }
    }, 'image/jpeg', 0.9);
  };

  const handleEnrol = async () => {
    if (!capturedBlob) return;
    setError('');
    setFeedback('Processing...');
    try {
      const file = new File([capturedBlob], 'face.jpg', { type: 'image/jpeg' });
      const result = await enrolUser.mutateAsync({ userId: selectedUserId, file });
      stopCamera();
      setStep('done');
      setFeedback(`Enrolled successfully! Quality: ${(result.quality_score * 100).toFixed(0)}%`);
      refetchStatus();
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Enrolment failed');
      setFeedback('');
    }
  };

  const reset = () => {
    stopCamera();
    setStep('select');
    setSelectedUserId('');
    setConsentGiven(false);
    setCapturedBlob(null);
    setFeedback('');
    setError('');
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Biometric Enrolment</h2>

      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-lg space-y-4">
        {step === 'select' && (
          <>
            <h3 className="font-medium text-white">Step 1: Select User</h3>
            <select
              value={selectedUserId}
              onChange={e => setSelectedUserId(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm"
            >
              <option value="">-- Select user --</option>
              {users.map((u: any) => (
                <option key={u.id} value={u.id}>{u.full_name} ({u.email})</option>
              ))}
            </select>
            {enrolmentStatus && (
              <div className={`text-sm p-2 rounded ${enrolmentStatus.is_enrolled ? 'bg-green-900/30 text-green-300' : 'bg-slate-700 text-slate-300'}`}>
                {enrolmentStatus.is_enrolled ? 'Already enrolled' : 'Not yet enrolled'}
              </div>
            )}
            <button
              onClick={() => setStep('consent')}
              disabled={!selectedUserId}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg py-2 text-sm"
            >
              Next: Consent
            </button>
          </>
        )}

        {step === 'consent' && (
          <>
            <h3 className="font-medium text-white">Step 2: Informed Consent</h3>
            <div className="bg-slate-700 rounded-lg p-3 text-sm text-slate-300">
              <p className="mb-2 font-medium">Biometric Data Collection Notice</p>
              <p>By proceeding, you consent to:</p>
              <ul className="list-disc ml-4 mt-1 space-y-1 text-slate-400">
                <li>Collection of your facial biometric data</li>
                <li>Storage of encrypted face embeddings</li>
                <li>Use for identity verification purposes only</li>
                <li>Deletion upon request</li>
              </ul>
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={consentGiven} onChange={e => setConsentGiven(e.target.checked)}
                className="w-4 h-4" />
              <span className="text-sm text-slate-300">I understand and give my consent</span>
            </label>
            {error && <p className="text-red-400 text-sm">{error}</p>}
            <div className="flex gap-2">
              <button onClick={handleConsentSubmit} disabled={!consentGiven || recordConsent.isPending}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg py-2 text-sm">
                {recordConsent.isPending ? 'Recording...' : 'Proceed to Capture'}
              </button>
              <button onClick={reset} className="bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-4 py-2 text-sm">Back</button>
            </div>
          </>
        )}

        {step === 'capture' && (
          <>
            <h3 className="font-medium text-white">Step 3: Capture Face</h3>
            <div className="relative bg-slate-900 rounded-lg overflow-hidden aspect-video">
              <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="border-2 border-blue-400 border-dashed rounded-full" style={{ width: '55%', height: '85%' }} />
              </div>
            </div>
            {feedback && <p className="text-green-400 text-sm">{feedback}</p>}
            {error && <p className="text-red-400 text-sm">{error}</p>}
            <div className="flex gap-2">
              <button onClick={capturePhoto}
                className="flex-1 flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg py-2 text-sm">
                <Camera size={16} />
                Capture
              </button>
              <button onClick={handleEnrol} disabled={!capturedBlob || enrolUser.isPending}
                className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg py-2 text-sm">
                <Upload size={16} />
                {enrolUser.isPending ? 'Enrolling...' : 'Enrol'}
              </button>
            </div>
          </>
        )}

        {step === 'done' && (
          <div className="text-center space-y-4">
            <CheckCircle className="text-green-400 mx-auto" size={48} />
            <h3 className="font-medium text-white">Enrolment Complete!</h3>
            {feedback && <p className="text-green-300 text-sm">{feedback}</p>}
            <button onClick={reset}
              className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-6 py-2 text-sm">
              Enrol Another User
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Enrolment;
