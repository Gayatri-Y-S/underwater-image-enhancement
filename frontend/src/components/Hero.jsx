function Hero({ summary }) {
  return (
    <section className="hero-card">
      <div>
        <p className="eyebrow">MTech Project Website</p>
        <h1>Underwater Image Enhancement using BAC Fusion</h1>
        <p className="hero-copy">
          Upload an underwater image, run the ResUNet + CBAM semantic branch, the physical
          reconstruction branch, and MuNet fusion, then inspect the enhanced output with PSNR,
          SSIM, and UIQM.
        </p>
      </div>
      <div className="hero-grid">
        <div className="metric-card">
          <span>Training Dataset</span>
          <strong>{summary?.training_plan?.train_dataset || "EUVP"}</strong>
          <small>80% training and 20% validation</small>
        </div>
        <div className="metric-card">
          <span>Testing Dataset</span>
          <strong>{summary?.training_plan?.test_dataset || "UIEB"}</strong>
          <small>Final evaluation dataset</small>
        </div>
        <div className="metric-card">
          <span>Fusion Formula</span>
          <strong>{summary?.training_plan?.fusion_formula || "μ·FST + (1−μ)·FSR"}</strong>
          <small>Balanced adaption compensation output</small>
        </div>
      </div>
    </section>
  );
}

export default Hero;
