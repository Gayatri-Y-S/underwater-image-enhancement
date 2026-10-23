import { useEffect, useMemo, useState } from "react";
import Hero from "./components/Hero";
import UploadPanel from "./components/UploadPanel";
import ResultPanel from "./components/ResultPanel";
import AuthPanel from "./components/AuthPanel";
import DashboardHeader from "./components/DashboardHeader";
import JobHistory from "./components/JobHistory";

const API_BASE = "http://127.0.0.1:8000/api";

function App() {
  const [summary, setSummary] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [user, setUser] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [authLoading, setAuthLoading] = useState(true);
  const [error, setError] = useState("");
  const [authError, setAuthError] = useState("");

  const api = useMemo(
    () => ({
      get: (path) => fetch(`${API_BASE}${path}`, { credentials: "include" }),
      post: async (path, body, isFormData = false) => {
        const headers = {};
        if (!isFormData) {
          headers["Content-Type"] = "application/json";
        }
        const csrfToken = getCookie("csrftoken");
        if (csrfToken) {
          headers["X-CSRFToken"] = csrfToken;
        }
        return fetch(`${API_BASE}${path}`, {
          method: "POST",
          credentials: "include",
          headers,
          body: isFormData ? body : JSON.stringify(body),
        });
      },
    }),
    []
  );

  useEffect(() => {
    api.get("/project-summary/")
      .then((response) => response.json())
      .then(setSummary)
      .catch(() => setSummary(null));
  }, [api]);

  useEffect(() => {
    bootstrapAuth();
  }, [api]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  async function bootstrapAuth() {
    setAuthLoading(true);
    try {
      await api.get("/auth/csrf/");
      const response = await api.get("/auth/profile/");
      if (!response.ok) {
        setUser(null);
        setJobs([]);
        return;
      }
      const payload = await response.json();
      setUser(payload.user);
      await loadJobs();
    } catch {
      setUser(null);
    } finally {
      setAuthLoading(false);
    }
  }

  async function loadJobs() {
    const response = await api.get("/jobs/");
    if (!response.ok) {
      setJobs([]);
      return;
    }
    const payload = await response.json();
    setJobs(payload);
  }

  async function handleAuthSubmit(formData) {
    setAuthError("");
    const path = authMode === "register" ? "/auth/register/" : "/auth/login/";
    const response = await api.post(path, formData);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Authentication failed.");
    }
    setUser(payload.user);
    setAuthMode("login");
    await loadJobs();
  }

  async function handleLogout() {
    setError("");
    const response = await api.post("/auth/logout/", {});
    if (response.ok) {
      setUser(null);
      setJobs([]);
      setResult(null);
      setSelectedFile(null);
      setPreviewUrl("");
    }
  }

  async function handleEnhance(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please choose an underwater image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    setLoading(true);
    setError("");

    try {
      const response = await api.post("/enhance/", formData, true);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || payload.message || "Enhancement failed.");
      }
      setResult(payload.job);
      await loadJobs();
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  if (authLoading) {
    return (
      <main className="app-shell">
        <Hero summary={summary} />
        <section className="workspace">
          <section className="panel">
            <div className="status-banner">Preparing secure session and loading your dashboard.</div>
          </section>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <Hero summary={summary} />
      {!user ? (
        <section className="workspace single-column">
          <AuthPanel
            mode={authMode}
            setMode={setAuthMode}
            onSubmit={handleAuthSubmit}
            authError={authError}
            setAuthError={setAuthError}
          />
        </section>
      ) : (
        <>
          <DashboardHeader user={user} onLogout={handleLogout} jobsCount={jobs.length} />
          <section className="workspace">
            <UploadPanel
              loading={loading}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              setSelectedFile={setSelectedFile}
              onSubmit={handleEnhance}
            />
            <ResultPanel result={result} previewUrl={previewUrl} loading={loading} error={error} />
          </section>
          <section className="workspace single-column">
            <JobHistory jobs={jobs} setResult={setResult} />
          </section>
        </>
      )}
    </main>
  );
}

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) {
    return parts.pop().split(";").shift();
  }
  return "";
}

export default App;
