import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, ShieldCheck, Users, UserCheck,
  History, AlertTriangle, BarChart2, Settings, Scan
} from 'lucide-react';

const links = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/verification', label: 'Verification', icon: Scan },
  { to: '/enrolment', label: 'Enrolment', icon: UserCheck },
  { to: '/users', label: 'Users', icon: Users },
  { to: '/history', label: 'History', icon: History },
  { to: '/security', label: 'Security', icon: AlertTriangle },
  { to: '/analytics', label: 'Analytics', icon: BarChart2 },
  { to: '/settings', label: 'Settings', icon: Settings },
];

const Sidebar: React.FC = () => {
  return (
    <div className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col">
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <ShieldCheck className="text-blue-400" size={24} />
          <div>
            <div className="font-bold text-sm">FaceVerification</div>
            <div className="text-xs text-slate-400">AI 2.0</div>
          </div>
        </div>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;
