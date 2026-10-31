import { Copy } from 'lucide-react';
import { toast } from '../store/toast';
import { Button } from './ui/Button';
import { Input } from './ui/form';

/** Read-only value with a copy button (for one-time secrets and links). */
export function CopyField({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex gap-2">
      <Input readOnly value={value} aria-label={label} className="font-mono text-xs" onFocus={(e) => e.target.select()} data-testid="copy-value" />
      <Button variant="outline" icon={<Copy className="h-4 w-4" />} onClick={() => void navigator.clipboard?.writeText(value).then(
        () => toast.success('Copied'), () => toast.error('Copy failed — select and copy manually'))}>Copy</Button>
    </div>
  );
}
