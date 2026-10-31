import type { LucideIcon } from 'lucide-react';
import {
  Activity, BarChart3, BellRing, Building2, Camera, ClipboardCheck, ClipboardList, Code2, FileCheck2, Gauge, History,
  LayoutDashboard, MonitorSmartphone, QrCode, ScanFace, Settings, ShieldAlert, UserPlus, Users,
} from 'lucide-react';
import type { Permission } from '../../lib/types';

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  any: Permission[];
}

export const NAV_SECTIONS: { title: string; items: NavItem[] }[] = [
  { title: 'Operate', items: [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard, any: ['dashboard:view'] },
    { to: '/verify', label: 'Live verification', icon: ScanFace, any: ['verification:run'] },
    { to: '/verify/qr', label: 'QR verification', icon: QrCode, any: ['verification:run'] },
    { to: '/enrol', label: 'Enrolment', icon: UserPlus, any: ['enrolment:capture'] },
    { to: '/documents', label: 'Document check', icon: FileCheck2, any: ['document:run'] },
    { to: '/history', label: 'History', icon: History, any: ['history:read_own', 'history:read_all'] },
  ] },
  { title: 'Review', items: [
    { to: '/reviews', label: 'Review queue', icon: ClipboardList, any: ['review:read'] },
    { to: '/approvals', label: 'Enrolment approvals', icon: ClipboardCheck, any: ['enrolment:read'] },
    { to: '/alerts', label: 'Alerts', icon: BellRing, any: ['security:read'] },
    { to: '/security', label: 'Security log', icon: ShieldAlert, any: ['security:read'] },
  ] },
  { title: 'Manage', items: [
    { to: '/users', label: 'Users', icon: Users, any: ['users:read'] },
    { to: '/organization', label: 'Organisation', icon: Building2, any: ['org:read'] },
    { to: '/cameras', label: 'Cameras', icon: Camera, any: ['cameras:read'] },
    { to: '/devices', label: 'Devices & kiosks', icon: MonitorSmartphone, any: ['devices:read'] },
    { to: '/developers', label: 'Developers', icon: Code2, any: ['integrations:manage'] },
  ] },
  { title: 'Insights', items: [
    { to: '/analytics', label: 'Analytics', icon: BarChart3, any: ['analytics:read'] },
    { to: '/models', label: 'Model performance', icon: Gauge, any: ['analytics:read'] },
    { to: '/system', label: 'System health', icon: Activity, any: ['system:health'] },
    { to: '/settings', label: 'Policy settings', icon: Settings, any: ['policy:read'] },
  ] },
];
