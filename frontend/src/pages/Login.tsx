import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';
import api from '../api/client';
import { useAuthStore } from '../store/authStore';

const Login: React.FC = () => {
  const [email, setEmail] = useState('admin@faceverification.local');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { setToken, setUser } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const loginResp = await api.post('/auth/login', { email, password });
      setToken(loginResp.data.access_token);
      const meResp = await api.get('/auth/me');
      setUser(meResp.data);
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-slate-800 rounded-xl border border-slate-700 p-8">
        <div className="flex items-center gap-3 mb-8">
          <ShieldCheck className="text-blue-400" size={32} />
          <div>
            <h1 className="text-xl font-bold text-white">FaceVerification AI</h1>
            <p className="text-slate-400 text-sm">Biometric Identity Platform 2.0</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-300 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm text-slate-300 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
              required
            />
          </div>
          {error && (
            <div className="bg-red-900/50 border border-red-700 rounded-lg px-3 py-2 text-red-300 text-sm">
              {error}
            </div>
          )}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-2 text-sm font-medium transition-colors disabled:opacity-50"
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className="text-center text-slate-500 text-xs mt-6">
          Default: admin@faceverification.local
        </p>
      </div>
    </div>
  );
};

export default Login;
