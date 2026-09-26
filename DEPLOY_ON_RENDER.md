# Deploy BioForm AI online with Render

This folder is ready for a Docker-based Render deployment.

## Before deployment

For a public demo, short exercise videos are recommended. The app performs
MediaPipe + OpenCV + PyTorch processing inside the web request, so long videos
can be slow on small cloud instances.

If you already have a trained `pinn_fatigue.pt` model on your Mac, copy it into:

`models/pinn_fatigue.pt`

before uploading the repository. That lets the online site use PINN prediction
immediately instead of starting with the biomechanical fatigue proxy.

## 1. Put this project on GitHub

Create a GitHub repository, for example:

`bioform-ai`

Upload the CONTENTS of this folder to the repository root. The repository root
must contain:

- app.py
- Dockerfile
- render.yaml
- requirements-deploy.txt
- backend/
- templates/
- static/
- exercise_config.json

Do not upload `.venv`.

## 2. Deploy on Render

1. Sign in to Render.
2. Choose New -> Blueprint.
3. Connect your GitHub account/repository.
4. Select the repository containing this `render.yaml`.
5. Approve the Blueprint and create the service.
6. Render will build the Docker image and start BioForm AI.
7. Open the generated `onrender.com` URL.

The included health-check path is:

`/health`

## 3. Public URL

Render generates a URL similar to:

`https://bioform-ai.onrender.com`

The exact subdomain depends on availability.

## 4. Custom domain (optional)

In the Render service dashboard, open Settings -> Custom Domains and add a
domain you own, for example:

`bioformai.in`

or

`bioformai.com`

Then apply the DNS records Render shows you.

## 5. Free-plan storage limitation

The free deployment is intended as a public demo. Generated session folders,
uploaded-derived results, newly trained models, and calibration changes are
runtime files and may disappear when a free instance restarts or spins down.

For persistent online training/session history, use a paid Render service with
a persistent disk and set:

`BIOFORM_STORAGE_ROOT=/var/data`

Mount the disk at `/var/data`.

## 6. Recommended college demo setup

For a reliable final-year-project demo:

- Keep your full training/validation version on your Mac.
- Put your already-trained `models/pinn_fatigue.pt` into this repository.
- Deploy the website online for public access.
- Use short, representative exercise clips online.
- Keep the local version as the source of truth for larger training runs and
  final scientific validation.
