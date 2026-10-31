import { lazy, Suspense, useEffect, type ReactNode } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { api, setUnauthorizedHandler } from './lib/api';
import { queryClient } from './lib/query';
import type { Me, Permission } from './lib/types';
import { useAuth } from './store/auth';
import { AppShell } from './components/layout/AppShell';
import { Toaster } from './components/ui/Toaster';
import { Spinner } from './components/ui/primitives';
import LoginPage from './pages/Login';
import { ForbiddenPage, NotFoundPage } from './pages/Errors';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Verify = lazy(() => import('./pages/Verify'));
const Enrol = lazy(() => import('./pages/Enrol'));
const Approvals = lazy(() => import('./pages/Approvals'));
const EnrolmentDetail = lazy(() => import('./pages/EnrolmentDetail'));
const Users = lazy(() => import('./pages/Users'));
const UserDetail = lazy(() => import('./pages/UserDetail'));
const History = lazy(() => import('./pages/History'));
const SessionDetail = lazy(() => import('./pages/SessionDetail'));
const Documents = lazy(() => import('./pages/Documents'));
const DocumentDetail = lazy(() => import('./pages/DocumentDetail'));
const Alerts = lazy(() => import('./pages/Alerts'));
const SecurityLog = lazy(() => import('./pages/SecurityLog'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Models = lazy(() => import('./pages/Models'));
const SystemHealth = lazy(() => import('./pages/SystemHealth'));
const SettingsPage = lazy(() => import('./pages/Settings'));
const Profile = lazy(() => import('./pages/Profile'));
const AccountSecurity = lazy(() => import('./pages/AccountSecurity'));
const Notifications = lazy(() => import('./pages/Notifications'));
const OrganizationPage = lazy(() => import('./pages/Organization'));
const InviteAccept = lazy(() => import('./pages/InviteAccept'));
const QrVerify = lazy(() => import('./pages/QrVerify'));
const MobileVerify = lazy(() => import('./pages/MobileVerify'));
const Cameras = lazy(() => import('./pages/Cameras'));
const Devices = lazy(() => import('./pages/Devices'));
const DeviceDetail = lazy(() => import('./pages/DeviceDetail'));
const Reviews = lazy(() => import('./pages/Reviews'));
const ReviewDetail = lazy(() => import('./pages/ReviewDetail'));
const Developers = lazy(() => import('./pages/Developers'));
const DeveloperClient = lazy(() => import('./pages/DeveloperClient'));
const ApiDocs = lazy(() => import('./pages/ApiDocs'));
const Kiosk = lazy(() => import('./pages/Kiosk'));

/** Pages that work without a central-server session (phone pairing links, the offline kiosk). */
const NO_SESSION_PATHS = ['/m/', '/kiosk'];

function Splash() {
  return (
    <div className="flex h-full items-center justify-center text-muted" role="status">
      <Spinner className="h-6 w-6" /><span className="sr-only">Loading</span>
    </div>
  );
}

/** Restore the session from HttpOnly cookies once at start-up. */
function SessionBootstrap({ children }: { children: ReactNode }) {
  const status = useAuth((s) => s.status);
  const setUser = useAuth((s) => s.setUser);
  const clear = useAuth((s) => s.clear);
  useEffect(() => {
    setUnauthorizedHandler(() => {
      clear();
      queryClient.clear();
    });
    let cancelled = false;
    if (NO_SESSION_PATHS.some((p) => window.location.pathname.startsWith(p))) {
      clear();
      return () => { cancelled = true; };
    }
    api.get<Me>('/auth/me')
      .then((r) => { if (!cancelled) setUser(r.data); })
      .catch(() => { if (!cancelled) clear(); });
    return () => { cancelled = true; };
  }, [setUser, clear]);
  if (status === 'unknown') return <Splash />;
  return <>{children}</>;
}

function RequireAuth({ children }: { children: ReactNode }) {
  const status = useAuth((s) => s.status);
  const location = useLocation();
  if (status !== 'authenticated') return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <>{children}</>;
}

function Guard({ any, children }: { any: Permission[]; children: ReactNode }) {
  const perms = useAuth((s) => s.user?.permissions ?? []);
  if (!any.some((p) => perms.includes(p))) return <ForbiddenPage />;
  return <Suspense fallback={<Splash />}>{children}</Suspense>;
}

function Home() {
  const perms = useAuth((s) => s.user?.permissions ?? []);
  if (perms.includes('dashboard:view')) return <Suspense fallback={<Splash />}><Dashboard /></Suspense>;
  return <Navigate to="/profile" replace />;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <SessionBootstrap>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/invite/:token" element={<Suspense fallback={<Splash />}><InviteAccept /></Suspense>} />
            <Route path="/m/:token" element={<Suspense fallback={<Splash />}><MobileVerify /></Suspense>} />
            <Route path="/kiosk" element={<Suspense fallback={<Splash />}><Kiosk /></Suspense>} />
            <Route element={<RequireAuth><AppShell /></RequireAuth>}>
              <Route index element={<Home />} />
              <Route path="verify" element={<Guard any={['verification:run']}><Verify /></Guard>} />
              <Route path="verify/qr" element={<Guard any={['verification:run']}><QrVerify /></Guard>} />
              <Route path="reviews" element={<Guard any={['review:read']}><Reviews /></Guard>} />
              <Route path="reviews/:reviewId" element={<Guard any={['review:read']}><ReviewDetail /></Guard>} />
              <Route path="organization" element={<Guard any={['org:read']}><OrganizationPage /></Guard>} />
              <Route path="cameras" element={<Guard any={['cameras:read']}><Cameras /></Guard>} />
              <Route path="devices" element={<Guard any={['devices:read']}><Devices /></Guard>} />
              <Route path="devices/:deviceId" element={<Guard any={['devices:read']}><DeviceDetail /></Guard>} />
              <Route path="developers" element={<Guard any={['integrations:manage']}><Developers /></Guard>} />
              <Route path="developers/docs" element={<Guard any={['integrations:manage']}><ApiDocs /></Guard>} />
              <Route path="developers/:clientId" element={<Guard any={['integrations:manage']}><DeveloperClient /></Guard>} />
              <Route path="notifications" element={<Suspense fallback={<Splash />}><Notifications /></Suspense>} />
              <Route path="account/security" element={<Suspense fallback={<Splash />}><AccountSecurity /></Suspense>} />
              <Route path="enrol" element={<Guard any={['enrolment:capture']}><Enrol /></Guard>} />
              <Route path="enrol/:userId" element={<Guard any={['enrolment:capture']}><Enrol /></Guard>} />
              <Route path="approvals" element={<Guard any={['enrolment:read']}><Approvals /></Guard>} />
              <Route path="enrolments/:userId" element={<Guard any={['enrolment:read']}><EnrolmentDetail /></Guard>} />
              <Route path="users" element={<Guard any={['users:read']}><Users /></Guard>} />
              <Route path="users/:userId" element={<Guard any={['users:read']}><UserDetail /></Guard>} />
              <Route path="history" element={<Guard any={['history:read_own', 'history:read_all']}><History /></Guard>} />
              <Route path="history/:sessionId" element={<Guard any={['history:read_own', 'history:read_all']}><SessionDetail /></Guard>} />
              <Route path="documents" element={<Guard any={['document:run']}><Documents /></Guard>} />
              <Route path="documents/:checkId" element={<Guard any={['document:run']}><DocumentDetail /></Guard>} />
              <Route path="alerts" element={<Guard any={['security:read']}><Alerts /></Guard>} />
              <Route path="security" element={<Guard any={['security:read']}><SecurityLog /></Guard>} />
              <Route path="analytics" element={<Guard any={['analytics:read']}><Analytics /></Guard>} />
              <Route path="models" element={<Guard any={['analytics:read']}><Models /></Guard>} />
              <Route path="system" element={<Guard any={['system:health']}><SystemHealth /></Guard>} />
              <Route path="settings" element={<Guard any={['policy:read']}><SettingsPage /></Guard>} />
              <Route path="profile" element={<Suspense fallback={<Splash />}><Profile /></Suspense>} />
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Routes>
        </SessionBootstrap>
        <Toaster />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
