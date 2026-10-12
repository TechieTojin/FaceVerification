import React from 'react';
import { Users, ShieldCheck, Zap, AlertTriangle, TrendingUp, Activity } from 'lucide-react';
import { useDashboard } from '../api/hooks';

const StatCard: React.FC<{ label: string; value: string | number; icon: React.ElementType; color: string }> = ({ label, value, icon: Icon, color }) => (
  <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
    <div className="flex items-center justify-between mb-2">
      <span className="text-slate-400 text-sm">{label}</span>
      <Icon size={18} className={color} />
    </div>
    <div className="text-2xl font-bold text-white">{value}</div>
  </div>
);

const Dashboard: React.FC = () => {
  const { data, isLoading, error } = useDashboard();

  if (isLoading) return <div className="text-slate-400">Loading dashboard...</div>;
  if (error) return <div className="text-red-400">Failed to load dashboard</div>;
  if (!data) return null;

  const outcomeColor = (outcome: string) => {
    switch (outcome) {
      case 'VERIFIED': return 'text-green-400';
      case 'REJECTED': return 'text-red-400';
      case 'SPOOF_DETECTED': return 'text-orange-400';
      case 'NO_FACE': return 'text-yellow-400';
      default: return 'text-slate-400';
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Dashboard Overview</h2>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Users" value={data.total_users} icon={Users} color="text-blue-400" />
        <StatCard label="Active Enrolments" value={data.active_enrolments} icon={ShieldCheck} color="text-green-400" />
        <StatCard label="Total Verifications" value={data.total_verifications} icon={Zap} color="text-purple-400" />
        <StatCard label="Success Rate" value={`${data.success_rate}%`} icon={TrendingUp} color="text-emerald-400" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatCard label="Successful" value={data.successful_verifications} icon={ShieldCheck} color="text-green-400" />
        <StatCard label="Failed" value={data.failed_verifications} icon={AlertTriangle} color="text-red-400" />
        <StatCard label="Spoof Attempts" value={data.spoof_attempts} icon={AlertTriangle} color="text-orange-400" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity size={18} className="text-blue-400" />
            <h3 className="font-medium text-white">Recent Events</h3>
          </div>
          {data.recent_events.length === 0 ? (
            <p className="text-slate-500 text-sm">No events yet</p>
          ) : (
            <div className="space-y-2">
              {data.recent_events.map((event: any) => (
                <div key={event.id} className="flex items-center justify-between py-1 border-b border-slate-700">
                  <span className={`text-sm font-medium ${outcomeColor(event.outcome)}`}>
                    {event.outcome}
                  </span>
                  <span className="text-xs text-slate-500">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h3 className="font-medium text-white mb-3">System Status</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Backend</span>
              <span className={`text-xs px-2 py-1 rounded ${data.backend_healthy ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                {data.backend_healthy ? 'Healthy' : 'Unhealthy'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">AI Models</span>
              <span className={`text-xs px-2 py-1 rounded ${data.models_ready ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'}`}>
                {data.models_ready ? 'Ready' : 'Loading'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Avg Latency</span>
              <span className="text-sm text-white">{data.avg_latency_ms.toFixed(0)}ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
