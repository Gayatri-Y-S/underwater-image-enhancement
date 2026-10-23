function JobHistory({ jobs, setResult }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <p className="eyebrow">History</p>
        <h2>Your enhancement jobs</h2>
      </div>

      {jobs.length ? (
        <div className="history-grid">
          {jobs.map((job) => (
            <article className="history-card" key={job.id}>
              <div>
                <strong>{job.original_name}</strong>
                <p>Status: {job.status}</p>
                <p>PSNR: {Number(job.psnr || 0).toFixed(3)}</p>
                <p>SSIM: {Number(job.ssim || 0).toFixed(4)}</p>
              </div>
              <button className="text-button" type="button" onClick={() => setResult(job)}>
                View result
              </button>
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <p>No enhancement jobs yet. Upload your first underwater image after login.</p>
        </div>
      )}
    </section>
  );
}

export default JobHistory;
