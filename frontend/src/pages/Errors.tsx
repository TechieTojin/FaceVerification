import { Link } from 'react-router-dom';
import { Compass, Lock } from 'lucide-react';
import { Card, EmptyState } from '../components/ui/primitives';

export function ForbiddenPage() {
  return (
    <Card>
      <EmptyState icon={<Lock className="h-5 w-5" />} title="You don’t have access to this page"
        body="Your role does not include the permission this page requires. Ask an administrator if you need it."
        action={<Link to="/" className="text-sm font-medium text-primary hover:underline">Go to start page</Link>} />
    </Card>
  );
}

export function NotFoundPage() {
  return (
    <Card>
      <EmptyState icon={<Compass className="h-5 w-5" />} title="Page not found"
        action={<Link to="/" className="text-sm font-medium text-primary hover:underline">Go to start page</Link>} />
    </Card>
  );
}
