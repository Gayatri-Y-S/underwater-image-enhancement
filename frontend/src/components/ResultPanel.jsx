function ResultPanel({ result, previewUrl, loading, error }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <p className="eyebrow">Output</p>
        <h2>Enhanced result and metrics</h2>
      </div>

      {error ? <div className="status-banner error">{error}</div> : null}
      {loading ? <div className="status-banner">Model is processing the image through FST, FSR, and MuNet.</div> : null}

      {result ? (
        <>
          <div className="comparison-grid">
            <figure>
              <img src={previewUrl || result.input_image_url} alt="Original underwater input" />
              <figcaption>Original input</figcaption>
            </figure>
            <figure>
              <img src={result.enhanced_image_url} alt="Enhanced underwater output" />
              <figcaption>Enhanced output</figcaption>
            </figure>
          </div>

          <div className="metrics-grid">
            <article className="metric-tile">
              <span>PSNR</span>
              <strong>{Number(result.psnr || 0).toFixed(3)}</strong>
            </article>
            <article className="metric-tile">
              <span>SSIM</span>
              <strong>{Number(result.ssim || 0).toFixed(4)}</strong>
            </article>
            <article className="metric-tile">
              <span>UIQM</span>
              <strong>{Number(result.uiqm || 0).toFixed(3)}</strong>
            </article>
            <article className="metric-tile">
              <span>μ</span>
              <strong>{Number(result.mu || 0).toFixed(4)}</strong>
            </article>
          </div>

          <div className="notes-card">
            <h3>Pipeline summary</h3>
            <p>{result.notes}</p>
          </div>

          <a className="primary-button download-button" href={result.enhanced_image_url} download>
            Download enhanced image
          </a>
        </>
      ) : (
        <div className="empty-state">
          <p>The enhanced image, PSNR, SSIM, UIQM, and download button will appear here after processing.</p>
        </div>
      )}
    </section>
  );
}

export default ResultPanel;
