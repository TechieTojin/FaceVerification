import React, { useState } from 'react';
import { Plus, Search, UserX } from 'lucide-react';
import { useUsers, useCreateUser, useDisableUser } from '../api/hooks';

const Users: React.FC = () => {
  const [search, setSearch] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ email: '', full_name: '', password: '', role: 'OPERATOR' });
  const [formError, setFormError] = useState('');

  const { data, isLoading } = useUsers(search || undefined);
  const createUser = useCreateUser();
  const disableUser = useDisableUser();

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');
    try {
      await createUser.mutateAsync(form);
      setShowCreate(false);
      setForm({ email: '', full_name: '', password: '', role: 'OPERATOR' });
    } catch (err: any) {
      setFormError(err.response?.data?.detail || 'Failed to create user');
    }
  };

  const handleDisable = async (id: string) => {
    if (confirm('Disable this user?')) {
      await disableUser.mutateAsync(id);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">User Management</h2>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2 text-sm"
        >
          <Plus size={16} />
          New User
        </button>
      </div>

      {showCreate && (
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h3 className="font-medium text-white mb-4">Create User</h3>
          <form onSubmit={handleCreate} className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Email</label>
              <input type="email" required value={form.email}
                onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Full Name</label>
              <input type="text" required value={form.full_name}
                onChange={e => setForm(f => ({ ...f, full_name: e.target.value }))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Password</label>
              <input type="password" required minLength={8} value={form.password}
                onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Role</label>
              <select value={form.role} onChange={e => setForm(f => ({ ...f, role: e.target.value }))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm">
                <option value="OPERATOR">OPERATOR</option>
                <option value="ADMIN">ADMIN</option>
                <option value="VIEWER">VIEWER</option>
              </select>
            </div>
            {formError && <div className="col-span-2 text-red-400 text-sm">{formError}</div>}
            <div className="col-span-2 flex gap-2">
              <button type="submit" disabled={createUser.isPending}
                className="bg-blue-600 hover:bg-blue-700 text-white rounded px-4 py-2 text-sm disabled:opacity-50">
                {createUser.isPending ? 'Creating...' : 'Create'}
              </button>
              <button type="button" onClick={() => setShowCreate(false)}
                className="bg-slate-700 hover:bg-slate-600 text-white rounded px-4 py-2 text-sm">
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="relative">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search users..."
          className="w-full pl-9 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
        />
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="border-b border-slate-700">
            <tr>
              {['Name', 'Email', 'Role', 'Status', 'Enrolled', 'Actions'].map(h => (
                <th key={h} className="text-left px-4 py-3 text-slate-400 font-medium text-xs">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {isLoading ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500">Loading...</td></tr>
            ) : data?.users?.map((user: any) => (
              <tr key={user.id} className="hover:bg-slate-750">
                <td className="px-4 py-3 text-white">{user.full_name}</td>
                <td className="px-4 py-3 text-slate-300">{user.email}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    user.role === 'ADMIN' ? 'bg-purple-900 text-purple-300' : 'bg-blue-900 text-blue-300'
                  }`}>{user.role}</span>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    user.is_active ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                  }`}>{user.is_active ? 'Active' : 'Disabled'}</span>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    user.has_enrolment ? 'bg-green-900 text-green-300' : 'bg-slate-700 text-slate-400'
                  }`}>{user.has_enrolment ? 'Yes' : 'No'}</span>
                </td>
                <td className="px-4 py-3">
                  <button onClick={() => handleDisable(user.id)} className="text-slate-400 hover:text-red-400 transition-colors" title="Disable">
                    <UserX size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {data && <div className="px-4 py-2 text-slate-500 text-xs border-t border-slate-700">{data.total} users total</div>}
      </div>
    </div>
  );
};

export default Users;
