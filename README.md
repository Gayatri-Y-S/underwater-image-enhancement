# Underwater Image Enhancement Website

This repository scaffolds an MTech project website for underwater image enhancement using:

- Frontend: HTML, CSS, JavaScript, React
- Backend: Django REST API
- Database: MySQL
- Authentication: Django session auth with React login and registration UI
- Model flow: ResUNet + CBAM for `FST`, physical model for `FSR`, MuNet for `μ`
- Fusion formula: `output = μ·FST + (1−μ)·FSR`
- Dataset plan: EUVP for training, UIEB for testing

## PPT analysis summary

The uploaded presentation describes a Balanced Adaption Compensation style system with these key requirements:

- User registration and login
- Upload underwater image
- Generate enhanced output
- Show PSNR and SSIM
- Allow downloading the enhanced result
- Use ResUNet + CBAM, physical reconstruction, and MuNet
- Train on EUVP with an 80/20 split
- Test on UIEB and report PSNR, SSIM, and UIQM

The current codebase implements the website, API contract, and a deterministic prototype inference pipeline that follows the same architecture. You can later replace the placeholder enhancement blocks with trained PyTorch weights.

## Project structure

```text
backend/
  config/                 Django project
  enhancer/               API, models, metrics, enhancement pipeline
  scripts/                dataset split, training, and evaluation entry points
frontend/
  src/                    React UI
docs/
```

## Auth flow

The project now includes:

- registration
- login
- logout
- authenticated dashboard
- user-linked enhancement jobs and history

Relevant files:

- `backend/enhancer/api_views.py`
- `backend/enhancer/models.py`
- `backend/enhancer/serializers.py`
- `frontend/src/components/AuthPanel.jsx`
- `frontend/src/components/DashboardHeader.jsx`
- `frontend/src/components/JobHistory.jsx`

## Backend setup

1. Create a virtual environment and install dependencies.
2. Copy `backend/.env.example` to `backend/.env` or set the environment variables manually.
3. Create a MySQL database named `underwater_enhancement`.
4. Run migrations.
5. Start Django.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

The React app runs on `http://localhost:5173` and expects the Django API on `http://127.0.0.1:8000`.

## Run order

After installing dependencies:

```powershell
cd backend
python manage.py migrate
python manage.py runserver
```

In another terminal:

```powershell
cd frontend
npm run dev
```

Then:

1. register a new user
2. login
3. upload an underwater image
4. inspect PSNR, SSIM, UIQM, and `μ`
5. download the enhanced image
6. view the saved result in job history

## Dataset and model workflow

### Training

- Use EUVP images under your dataset root.
- Split: 80% train, 20% validation.
- Learn:
  - `FST` with ResUNet + CBAM
  - `FSR` with the physical reconstruction branch
  - `μ` with MuNet

```powershell
cd backend
python scripts\train_model.py --euvp-root "C:\path\to\EUVP"
```

### Testing

- Use UIEB as the held-out testing dataset.
- Final output: `μ·FST + (1−μ)·FSR`
- Evaluate using PSNR, SSIM, and UIQM

```powershell
cd backend
python scripts\evaluate_model.py --uieb-root "C:\path\to\UIEB"
```

## Notes

- The shipped inference pipeline is a prototype so the site works immediately.
- To convert this into your final thesis model, replace the placeholder logic in `backend/enhancer/pipeline.py` with trained PyTorch checkpoints and dataloaders.

## Real training code

The repository now also includes trainable BAC model code:

- `backend/enhancer/networks.py`
  - ResUNet + CBAM semantic branch for `FST`
  - learnable physical branch for `FSR`
  - MuNet for adaptive fusion weight
- `backend/enhancer/training.py`
  - fused loss, branch loss, PSNR, SSIM, and UIQM tracking
- `backend/enhancer/dataset.py`
  - paired-image discovery for common EUVP and UIEB folder layouts
- `backend/scripts/train_bac_model.py`
  - full EUVP training loop with checkpoint saving
- `backend/scripts/evaluate_bac_model.py`
  - UIEB evaluation using saved weights

Example commands:

```powershell
cd backend
python scripts\train_bac_model.py --euvp-root "C:\path\to\EUVP" --epochs 100 --batch-size 4
python scripts\evaluate_bac_model.py --uieb-root "C:\path\to\UIEB"
```

When `backend/weights/bac_model.pt` exists, the Django API automatically loads it and uses the trained model for website inference.
