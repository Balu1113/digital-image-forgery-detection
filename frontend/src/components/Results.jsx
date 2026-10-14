export default function Results({ result, onReset }) {
  if (!result) return null;

  const isReal = result.label === 'Real';
  const confidence = Number(result.confidence ?? 0);
  const forgeryProb = Number(result.forgery_probability ?? 0);

  return (
    <section className="card results">
      <h2>Analysis result</h2>

      <div className={`verdict ${isReal ? 'verdict-real' : 'verdict-doctored'}`}>
        <div className="verdict-label">{isReal ? '✓ Authentic image' : '✗ Forgery detected'}</div>
        <div className="confidence-bar">
          <div className="confidence-fill" style={{ width: `${Math.min(100, confidence)}%` }} />
        </div>
        <p className="confidence-text">
          Confidence: {confidence.toFixed(2)}% &middot; Forgery probability: {(forgeryProb * 100).toFixed(2)}%
        </p>
      </div>

      <div className="images">
        <figure>
          <figcaption>Original</figcaption>
          <img src={result.original_image_url} alt="Uploaded original" />
        </figure>
        <figure>
          <figcaption>ELA (error level analysis)</figcaption>
          <img src={result.ela_image_url} alt="ELA visualisation" />
        </figure>
      </div>

      <p className="muted small">
        The model scores the ELA image. Bright, textured regions can indicate manipulation, but
        heavily compressed photos may also show strong ELA.
      </p>

      <button className="btn" onClick={onReset}>
        Analyze another image
      </button>
    </section>
  );
}