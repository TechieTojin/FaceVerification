import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './store/authStore';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Verification from './pages/Verification';
import Users from './pages/Users';
import Enrolment from './pages/Enrolment';
import History from './pages/History';
import Security from './pages/Security';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30000 },
  },
});

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
};

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="verification" element={<Verification />} />
            <Route path="enrolment" element={<Enrolment />} />
            <Route path="users" element={<Users />} />
            <Route path="history" element={<History />} />
            <Route path="security" element={<Security />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
