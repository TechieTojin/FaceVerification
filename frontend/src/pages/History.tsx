import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useVerificationEvents } from '../api/hooks';
import type { OutcomeType } from '../types';

const outcomeColors: Record<OutcomeType, string> = {
  VERIFIED: 'text-green-400',
  REJECTED: 'text-red-400',
  NO_FACE: 'text-yellow-400',
  MULTIPLE_FACES: 'text-orange-400',
  SPOOF_DETECTED: 'text-red-500',
  ERROR: 'text-slate-400',
  TIMEOUT: 'text-slate-400',
  LOCKOUT: 'text-red-500',
};

const History: React.FC = () => {
  const [page, setPage] = useState(1);
  const [outcomeFilter, setOutcomeFilter] = useState('');
  const { data, isLoading } = useVerificationEvents(page, undefined, outcomeFilter || undefined);

  const totalPages = data ? Math.ceil(data.total / 20) : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Verification History</h2>
        <select
          value={outcomeFilter}
          onChange={e => { setOutcomeFilter(e.target.value); setPage(1); }}
          className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
        >
          <option value="">All outcomes</option>
          {['VERIFIED', 'REJECTED', 'NO_FACE', 'MULTIPLE_FACES', 'SPOOF_DETECTED', 'ERROR'].map(o => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="border-b border-slate-700">
            <tr>
              {['Timestamp', 'Outcome', 'Distance', 'Latency', 'Liveness', 'Explanation'].map(h => (
                <th key={h} className="text-left px-4 py-3 text-slate-400 font-medium text-xs">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {isLoading ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500">Loading...</td></tr>
            ) : data?.events?.length === 0 ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500">No events yet</td></tr>
            ) : data?.events?.map((event: any) => (
              <tr key={event.id} className="hover:bg-slate-750">
                <td className="px-4 py-3 text-slate-400 text-xs">
                  {new Date(event.timestamp).toLocaleString()}
                </td>
                <td className="px-4 py-3">
                  <span className={`font-medium ${outcomeColors[event.outcome as OutcomeType]}`}>
                    {event.outcome}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {event.distance !== null ? event.distance?.toFixed(3) : '-'}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {event.inference_ms ? `${event.inference_ms.toFixed(0)}ms` : '-'}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {event.liveness_score !== null ? event.liveness_score?.toFixed(2) : '-'}
                </td>
                <td className="px-4 py-3 text-slate-400 text-xs max-w-xs truncate">
                  {event.explanation || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {data && (
          <div className="px-4 py-3 border-t border-slate-700 flex items-center justify-between text-sm">
            <span className="text-slate-500">{data.total} events total</span>
            <div className="flex items-center gap-2">
              <button onClick={() => setPage(p => Math.max(1, p-1))} disabled={page === 1}
                className="text-slate-400 hover:text-white disabled:opacity-30">
                <ChevronLeft size={16} />
              </button>
              <span className="text-slate-300">Page {page} of {totalPages || 1}</span>
              <button onClick={() => setPage(p => Math.min(totalPages, p+1))} disabled={page >= totalPages}
                className="text-slate-400 hover:text-white disabled:opacity-30">
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
