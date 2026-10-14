import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../auth/AuthContext';
import HistoryList from '../components/HistoryList';
import Results from '../components/Results';
import UploadPanel from '../components/UploadPanel';

export default function HomePage() {
  const { user, isAuthenticated } = useAuth();

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((data) => !cancelled && setHealth(data))
      .catch(() => !cancelled && setHealth({ inference: { reachable: false } }));
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshHistory = useCallback(async () => {
    if (!api.getToken()) {
      setHistory([]);
      return;
    }
    setHistoryLoading(true);
    try {
      setHistory(await api.history());
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      refreshHistory();
    } else {
      setHistory([]);
    }
  }, [isAuthenticated, refreshHistory]);

  async function handleAnalyze(file) {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await api.analyze(file);
      setResult(data);
      if (api.getToken()) refreshHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id) {
    try {
      await api.deletePrediction(id);
      setHistory((items) => items.filter((item) => item.id !== id));
    } catch (err) {
      setError(err.message);
    }
  }

  const inferenceOk = health?.inference?.reachable;

  return (
    <>
      <header className="hero">
        <h1>🔍 Image Forgery Detection</h1>
        <p className="subtitle">Error Level Analysis + a fusion of InceptionV3 &amp; VGG16</p>
        <div className={`status ${inferenceOk ? 'status-ok' : 'status-bad'}`}>
          {inferenceOk ? 'Inference service online' : 'Inference service offline'}
        </div>
      </header>

      {error && (
        <div className="alert">
          <span>{error}</span>
          <button className="btn ghost small" onClick={() => setError('')}>
            ✕
          </button>
        </div>
      )}

      <main className="layout">
        <div className="column">
          {isAuthenticated ? (
            <>
              <section className="card">
                <UploadPanel onAnalyze={handleAnalyze} disabled={loading} />
                {loading && (
                  <div className="loading">
                    <div className="spinner" />
                    <p>Analyzing image…</p>
                  </div>
                )}
              </section>

              <Results result={result} onReset={() => setResult(null)} />
            </>
          ) : (
            <section className="card locked">
              <h2>Sign in to analyze images</h2>
              <p className="muted">
                Uploading and analyzing images requires an account. Log in or create one to get
                started.
              </p>
              <div className="row">
                <Link className="btn" to="/login">
                  Log in
                </Link>
                <Link className="btn ghost" to="/register">
                  Register
                </Link>
              </div>
            </section>
          )}
        </div>

        <div className="column">
          <HistoryList
            items={history}
            loading={historyLoading}
            authenticated={Boolean(user)}
            onDelete={handleDelete}
            onRefresh={refreshHistory}
          />
        </div>
      </main>
    </>
  );
}