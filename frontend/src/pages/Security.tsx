import React from 'react';
import { Shield } from 'lucide-react';
import { useSecurityEvents } from '../api/hooks';

const severityColors: Record<string, string> = {
  HIGH: 'text-red-400 bg-red-900/30',
  WARNING: 'text-yellow-400 bg-yellow-900/30',
  INFO: 'text-blue-400 bg-blue-900/30',
};

const Security: React.FC = () => {
  const { data, isLoading } = useSecurityEvents();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Shield className="text-orange-400" size={20} />
        <h2 className="text-lg font-semibold text-white">Security Events</h2>
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="border-b border-slate-700">
            <tr>
              {['Timestamp', 'Event Type', 'Severity', 'Description', 'User'].map(h => (
                <th key={h} className="text-left px-4 py-3 text-slate-400 font-medium text-xs">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {isLoading ? (
              <tr><td colSpan={5} className="px-4 py-8 text-center text-slate-500">Loading...</td></tr>
            ) : data?.events?.length === 0 ? (
              <tr><td colSpan={5} className="px-4 py-8 text-center text-slate-500">No security events</td></tr>
            ) : data?.events?.map((event: any) => (
              <tr key={event.id}>
                <td className="px-4 py-3 text-slate-400 text-xs">
                  {new Date(event.timestamp).toLocaleString()}
                </td>
                <td className="px-4 py-3 text-white font-medium">{event.event_type}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${severityColors[event.severity] || 'text-slate-400'}`}>
                    {event.severity}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-300 text-xs max-w-xs truncate">{event.description}</td>
                <td className="px-4 py-3 text-slate-400 text-xs">{event.user_id?.slice(0, 8) || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data && <div className="px-4 py-2 text-slate-500 text-xs border-t border-slate-700">{data.total} events total</div>}
      </div>
    </div>
  );
};

export default Security;
