import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { useAnalytics } from '../api/hooks';

const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#f97316', '#64748b'];

const Analytics: React.FC = () => {
  const { data, isLoading } = useAnalytics();

  if (isLoading) return <div className="text-slate-400">Loading analytics...</div>;
  if (!data) return <div className="text-slate-400">No data</div>;

  const outcomeData = Object.entries(data.outcome_distribution || {}).map(([name, value]) => ({ name, value }));
  const latencyData = Object.entries(data.latency_histogram || {}).map(([name, value]) => ({ name, value }));
  const dailyData = (data.verifications_by_day || []).map((d: any) => ({
    date: d.date.slice(5),
    count: d.count,
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Analytics</h2>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h3 className="font-medium text-white mb-4">Verifications per Day (last 30 days)</h3>
          {dailyData.length === 0 ? (
            <p className="text-slate-500 text-sm">No data yet</p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={dailyData}>
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
                <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h3 className="font-medium text-white mb-4">Outcome Distribution</h3>
          {outcomeData.length === 0 ? (
            <p className="text-slate-500 text-sm">No data yet</p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={outcomeData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name }) => name}>
                  {outcomeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h3 className="font-medium text-white mb-4">Latency Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={latencyData}>
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h3 className="font-medium text-white mb-4">Failure Reasons</h3>
          {Object.keys(data.failure_reasons || {}).length === 0 ? (
            <p className="text-slate-500 text-sm">No failures recorded</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(data.failure_reasons).map(([reason, count]: [string, any]) => (
                <div key={reason} className="flex items-center justify-between">
                  <span className="text-sm text-slate-300">{reason}</span>
                  <span className="text-sm text-white font-medium">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analytics;
