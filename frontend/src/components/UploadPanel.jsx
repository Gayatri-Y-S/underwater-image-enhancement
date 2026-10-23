function UploadPanel({ loading, selectedFile, previewUrl, setSelectedFile, onSubmit }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <p className="eyebrow">Input</p>
        <h2>Upload underwater image</h2>
      </div>

      <form className="upload-form" onSubmit={onSubmit}>
        <label className="upload-dropzone">
          <input
            type="file"
            accept=".jpg,.jpeg,.png"
            onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
          />
          <span>{selectedFile ? selectedFile.name : "Choose JPG or PNG image"}</span>
          <small>Frontend: React + HTML + CSS + JavaScript</small>
        </label>

        {previewUrl ? <img className="preview-image" src={previewUrl} alt="Original preview" /> : null}

        <button className="primary-button" type="submit" disabled={loading}>
          {loading ? "Enhancing image..." : "Run enhancement"}
        </button>
      </form>
    </section>
  );
}

export default UploadPanel;
