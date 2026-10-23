import { useState } from "react";

function AuthPanel({ mode, setMode, onSubmit, authError, setAuthError }) {
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    confirm_password: "",
  });
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setAuthError("");
    try {
      const payload =
        mode === "register"
          ? form
          : {
              username: form.username,
              password: form.password,
            };
      await onSubmit(payload);
    } catch (error) {
      setAuthError(error.message);
    } finally {
      setSubmitting(false);
    }
  }

  function updateField(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  return (
    <section className="panel auth-panel">
      <div className="panel-heading">
        <p className="eyebrow">Secure Access</p>
        <h2>{mode === "register" ? "Create your project account" : "Login to your dashboard"}</h2>
      </div>

      <form className="auth-form" onSubmit={handleSubmit}>
        <label className="field-group">
          <span>Username</span>
          <input name="username" value={form.username} onChange={updateField} required />
        </label>

        {mode === "register" ? (
          <label className="field-group">
            <span>Email</span>
            <input name="email" type="email" value={form.email} onChange={updateField} required />
          </label>
        ) : null}

        <label className="field-group">
          <span>Password</span>
          <input name="password" type="password" value={form.password} onChange={updateField} required />
        </label>

        {mode === "register" ? (
          <label className="field-group">
            <span>Confirm password</span>
            <input
              name="confirm_password"
              type="password"
              value={form.confirm_password}
              onChange={updateField}
              required
            />
          </label>
        ) : null}

        {authError ? <div className="status-banner error">{authError}</div> : null}

        <button className="primary-button" type="submit" disabled={submitting}>
          {submitting ? "Submitting..." : mode === "register" ? "Register account" : "Login"}
        </button>
      </form>

      <div className="auth-switch">
        <span>{mode === "register" ? "Already have an account?" : "New user?"}</span>
        <button className="text-button" type="button" onClick={() => setMode(mode === "register" ? "login" : "register")}>
          {mode === "register" ? "Go to login" : "Create account"}
        </button>
      </div>
    </section>
  );
}

export default AuthPanel;
