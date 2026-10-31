import { Badge } from './ui/primitives';
import { outcomeMeta } from '../lib/format';

export function OutcomeBadge({ outcome, active }: { outcome: string | null | undefined; active?: boolean }) {
  if (active) return <Badge tone="info">In progress</Badge>;
  const m = outcomeMeta(outcome);
  return <Badge tone={m.tone}>{m.label}</Badge>;
}
