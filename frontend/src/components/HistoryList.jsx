import { useState } from 'react';
import { Link } from 'react-router-dom';

export default function HistoryList({ items, loading, onDelete, onRefresh, authenticated, onFeedback }) {
  const [busyId, setBusyId] = useState(null);

  if (!authenticated) {
    return (
      <section className="card">
        <h2>History</h2>
        <p className="muted">
          <Link to="/login">Log in</Link> or <Link to="/register">register</Link> to save and
          review your previous analyses.
        </p>
      </section>
    );
  }

  async function sendFeedback(id, verdict) {
    setBusyId(id);
    await onFeedback?.(id, verdict);
    setBusyId(null);
  }

  return (
    <section className="card">
      <div className="row space-between">
        <h2>History</h2>
        <button className="btn ghost small" onClick={onRefresh} disabled={loading}>
          Refresh
        </button>
      </div>

      {loading && <p className="muted">Loading…</p>}
      {!loading && items.length === 0 && <p className="muted">No analyses yet.</p>}

      <ul className="history">
        {items.map((item) => (
          <li key={item.id}>
            <img src={item.ela_image_url} alt="" />
            <div className="history-meta">
              <span className={`tag ${item.label === 'Real' ? 'tag-real' : 'tag-doctored'}`}>
                {item.label}
              </span>
              <span className="history-file" title={item.filename}>
                {item.filename || 'image'}
              </span>
              <span className="muted small">
                {Number(item.confidence).toFixed(1)}% &middot; {new Date(item.created_at).toLocaleString()}
              </span>
            </div>
            {!item.feedback || item.feedback === 'pending' ? (
              <span className="history-feedback">
                <button
                  className="btn ghost small"
                  onClick={() => sendFeedback(item.id, 'correct')}
                  disabled={busyId === item.id}
                  aria-label={`Mark verdict for ${item.filename || 'image'} correct`}
                  title="Verdict correct"
                >
                  ✓
                </button>
                <button
                  className="btn ghost small"
                  onClick={() => sendFeedback(item.id, 'incorrect')}
                  disabled={busyId === item.id}
                  aria-label={`Mark verdict for ${item.filename || 'image'} incorrect`}
                  title="Verdict incorrect"
                >
                  ✗
                </button>
              </span>
            ) : (
              <span
                className={`feedback-badge ${
                  item.feedback === 'correct' ? 'feedback-correct' : 'feedback-incorrect'
                }`}
                title={`Feedback: ${item.feedback}`}
              >
                {item.feedback === 'correct' ? '✓' : '✗'}
              </span>
            )}
            <button className="btn ghost small" onClick={() => onDelete(item.id)} aria-label="Delete">
              ✕
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
