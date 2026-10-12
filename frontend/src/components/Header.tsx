import React from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, User } from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import api from '../api/client';

const Header: React.FC = () => {
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await api.post('/auth/logout');
    } catch {}
    logout();
    navigate('/login');
  };

  return (
    <header className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-6">
      <h1 className="text-sm font-medium text-slate-300">
        Biometric Identity Verification Platform
      </h1>
      <div className="flex items-center gap-4">
        {user && (
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <User size={16} />
            <span>{user.email}</span>
            <span className="px-2 py-0.5 bg-blue-900 text-blue-300 rounded text-xs">{user.role}</span>
          </div>
        )}
        <button
          onClick={handleLogout}
          className="flex items-center gap-1 text-slate-400 hover:text-white transition-colors text-sm"
        >
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </header>
  );
};

export default Header;
