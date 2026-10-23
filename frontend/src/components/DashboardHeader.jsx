function DashboardHeader({ user, onLogout, jobsCount }) {
  return (
    <section className="hero-card dashboard-header">
      <div>
        <p className="eyebrow">User Dashboard</p>
        <h2>{user.username}'s underwater enhancement workspace</h2>
        <p className="hero-copy">
          Your uploads, metrics, and downloadable results stay attached to your account so the project
          behaves like a complete authenticated application.
        </p>
      </div>
      <div className="dashboard-actions">
        <div className="metric-card">
          <span>Account</span>
          <strong>{user.email || user.username}</strong>
          <small>Authenticated session active</small>
        </div>
        <div className="metric-card">
          <span>Jobs</span>
          <strong>{jobsCount}</strong>
          <small>Saved in your history</small>
        </div>
        <button className="secondary-button" type="button" onClick={onLogout}>
          Logout
        </button>
      </div>
    </section>
  );
}

export default DashboardHeader;
