import React, { useState, useEffect } from 'react';
import { Save, Settings as SettingsIcon } from 'lucide-react';
import { useSettings, useUpdateSettings } from '../api/hooks';

const Settings: React.FC = () => {
  const { data: settings, isLoading } = useSettings();
  const updateSettings = useUpdateSettings();
  const [form, setForm] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (settings) {
      const initial: Record<string, string> = {};
      Object.entries(settings).forEach(([key, val]: [string, any]) => {
        initial[key] = val.value;
      });
      setForm(initial);
    }
  }, [settings]);

  const handleSave = async () => {
    try {
      await updateSettings.mutateAsync(form);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e) {
      console.error('Settings save failed:', e);
    }
  };

  if (isLoading) return <div className="text-slate-400">Loading settings...</div>;

  const settingLabels: Record<string, string> = {
    max_face_distance: 'Max Face Distance (0-1)',
    min_face_quality: 'Min Face Quality (0-1)',
    enable_liveness: 'Enable Liveness Check (true/false)',
    session_timeout_seconds: 'Session Timeout (seconds)',
    max_attempts_per_session: 'Max Attempts per Session',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <SettingsIcon className="text-slate-400" size={20} />
        <h2 className="text-lg font-semibold text-white">System Settings</h2>
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-lg space-y-4">
        <p className="text-slate-400 text-sm">Configure verification policy settings. Changes take effect immediately.</p>

        {Object.entries(form).map(([key, value]) => (
          <div key={key}>
            <label className="block text-sm text-slate-300 mb-1">{settingLabels[key] || key}</label>
            {settings?.[key]?.description && (
              <p className="text-xs text-slate-500 mb-1">{settings[key].description}</p>
            )}
            <input
              type="text"
              value={value}
              onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
            />
          </div>
        ))}

        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleSave}
            disabled={updateSettings.isPending}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg px-4 py-2 text-sm"
          >
            <Save size={16} />
            {updateSettings.isPending ? 'Saving...' : 'Save Settings'}
          </button>
          {saved && <span className="text-green-400 text-sm">Settings saved!</span>}
        </div>
      </div>
    </div>
  );
};

export default Settings;
