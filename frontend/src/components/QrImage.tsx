import { useEffect, useState } from 'react';
import QRCode from 'qrcode';
import { cn } from '../lib/cn';

/** Renders a QR code locally (no third-party service ever sees the value). */
export function QrImage({ value, size = 220, label, className }: { value: string; size?: number; label: string; className?: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    QRCode.toDataURL(value, { width: size, margin: 1, errorCorrectionLevel: 'M', color: { dark: '#0b1220', light: '#ffffff' } })
      .then((url) => { if (alive) setSrc(url); })
      .catch(() => { if (alive) setSrc(null); });
    return () => { alive = false; };
  }, [value, size]);
  return src
    ? <img src={src} width={size} height={size} alt={label} className={cn('rounded-lg bg-white p-2', className)} data-testid="qr-image" />
    : <div style={{ width: size, height: size }} className={cn('animate-pulse rounded-lg bg-surface-2', className)} aria-label={label} />;
}
