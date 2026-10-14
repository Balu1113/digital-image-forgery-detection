import { useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import AuthCard from '../components/AuthCard';

export default function LoginPage() {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || '/app';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  if (isAuthenticated) {
    return <Navigate to={from} replace />;
  }

  async function onSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      await login(username, password);
      navigate(from, { replace: true });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthCard
      title="Log in"
      subtitle="Sign in to save and review your analysis history."
      footer={
        <>
          Don&apos;t have an account? <Link to="/register">Create one</Link>
        </>
      }
    >
      <form className="auth-form-vertical" onSubmit={onSubmit}>
        <label>
          Username
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
            autoFocus
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>

        {error && <p className="error-text">{error}</p>}

        <button className="btn" type="submit" disabled={busy}>
          {busy ? 'Signing in…' : 'Log in'}
        </button>
      </form>
    </AuthCard>
  );
}
