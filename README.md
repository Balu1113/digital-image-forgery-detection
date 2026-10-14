# Digital Image Forgery Detection

An end-to-end deep learning application that detects tampered ("doctored") images
using **Error Level Analysis (ELA)** and a **fusion of InceptionV3 + VGG16**.

The system is split into three services:

| Service  | Stack              | Port | Responsibility                                        |
|----------|--------------------|------|-------------------------------------------------------|
| Frontend | React + Vite       | 5173 | UI: upload, results, auth, history                    |
| Backend  | Django + DRF       | 8000 | API gateway: users, prediction history, admin, proxying |
| Inference| Flask + TensorFlow | 5000 | Owns the model; exposes `POST /predict`               |

```
React (5173)  ──HTTP──►  Django REST (8000)  ──HTTP──►  Flask inference (5000)
                              │                              │
                              ▼                              ▼
                         SQLite DB + media            image_real_or_fake.keras
```

The browser only ever talks to Django. Django calls Flask for inference and
persists each result (label, confidence, original + ELA images) so users can
review their history.

---

## How the model works

1. **Error Level Analysis** – the image is re-saved as JPEG at quality 90 and the
   pixel-wise difference against the original is computed. Edited regions have a
   different compression history and show up as bright "hot spots".
2. **Feature extraction** – the ELA image is resized to `224x224` and passed
   through two frozen ImageNet backbones: InceptionV3 and VGG16.
3. **Fusion head** – both feature maps are concatenated and passed through dense
   layers that output a single sigmoid probability.
4. **Decision** – the sigmoid is `P(real)`. If `P(real) >= threshold`
   (default `0.45`) the image is labelled **Real**, otherwise **Doctored**;
   `forgery_probability = 1 - P(real)`.

> Why `P(real)` and not `P(doctored)`? `flow_from_directory` assigns label `1` to
> the alphabetically-last class folder. With folders named `Doctored` and `Real`,
> that makes `Real = 1`, so the network's sigmoid output is `P(real)`.

## Project layout

```
.
├── app.py                  # Flask inference microservice
├── config.py               # Flask configuration
├── model_utils.py          # ELA + inference utilities
├── ifdmodel.py             # Fusion model architecture
├── train.py                # Training pipeline
├── image_real_or_fake.keras
├── requirements.txt        # Flask deps
├── tests/                  # Flask pytest suite
│
├── backend/                # Django API gateway
│   ├── manage.py
│   ├── config/             # settings / urls / wsgi / asgi
│   ├── predictions/        # models, serializers, views, services, admin
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/               # React + Vite SPA
│   ├── index.html
│   ├── vite.config.js      # dev proxy -> Django
│   ├── src/
│   └── .env.example
│
├── docker-compose.yml      # runs all three services
├── Dockerfile              # Flask image
└── .env.example
```

## Quick start (local development)

Create a virtual environment once, then install both Python requirement sets:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -r requirements.txt
pip install -r backend/requirements.txt
```

Run each service in its own terminal:

```bash
# 1) Flask inference service  -> http://127.0.0.1:5000
python app.py

# 2) Django API gateway       -> http://127.0.0.1:8000
cd backend
python manage.py migrate
python manage.py runserver 127.0.0.1:8000

# 3) React frontend           -> http://127.0.0.1:5173
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api`, `/media` and
`/admin` to Django, so no CORS setup is needed locally.

The SPA has three routes (React Router):

| Route       | Page                                                         |
|-------------|--------------------------------------------------------------|
| `/`         | Analysis dashboard: upload, verdict, ELA comparison, history |
| `/login`    | Log in (obtains an auth token)                               |
| `/register` | Create an account                                            |

Optional: create an admin login with `python manage.py createsuperuser` (from
`backend/`) and visit <http://127.0.0.1:8000/admin/> to browse predictions.

> The model is ~180 MB and TensorFlow loads it lazily on the first prediction, so
> the first request can take a few seconds.

## API reference

### Flask inference service (`:5000`)

| Method | Path           | Description                          |
|--------|----------------|--------------------------------------|
| `GET`  | `/`            | Service metadata                     |
| `GET`  | `/health`      | Liveness + model status              |
| `POST` | `/predict`     | Analyse an image (`file` form field) |
| `POST` | `/api/predict` | Alias of `/predict`                  |

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:5000/predict
```

```json
{
  "prediction": "Doctored",
  "confidence": 92.31,
  "forgery_probability": 0.9231,
  "real_probability": 0.0769,
  "filename": "photo.jpg",
  "original_image": "<base64 JPEG>",
  "ela_image": "<base64 JPEG>"
}
```

### Django API gateway (`:8000`, base path `/api`)

| Method   | Path                  | Auth | Description                            |
|----------|-----------------------|------|----------------------------------------|
| `POST`   | `/api/auth/register/` | –    | Create account, returns a token        |
| `POST`   | `/api/auth/login/`    | –    | Obtain a token                         |
| `POST`   | `/api/auth/logout/`   | ✓    | Invalidate the token                   |
| `GET`    | `/api/auth/me/`       | ✓    | Current user                           |
| `GET`    | `/api/health/`        | –    | Django status + inference reachability |
| `POST`   | `/api/predictions/`   | ✓    | Upload an image, store the result      |
| `GET`    | `/api/predictions/`   | ✓    | List the current user's history        |
| `GET`/`DELETE` | `/api/predictions/<id>/` | ✓ | Retrieve / delete one prediction  |

Authentication uses DRF token auth: send `Authorization: Token <key>`.
Analyzing images requires an account — logged-out visitors see a sign-in prompt
instead of the upload panel.

## Configuration

**Flask** (`.env.example`): `MODEL_PATH`, `ELA_QUALITY`,
`CLASSIFICATION_THRESHOLD` (default `0.45`), `MAX_CONTENT_LENGTH`, `HOST`, `PORT`,
`FLASK_DEBUG`.

**Django** (`backend/.env.example`): `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`,
`DJANGO_ALLOWED_HOSTS`, `DJANGO_DB_PATH`, `CORS_ALLOWED_ORIGINS`,
`FORGERY_API_URL`, `FORGERY_API_TIMEOUT`, `MAX_UPLOAD_SIZE`.

**Frontend** (`frontend/.env.example`): `VITE_API_URL` (empty uses the dev proxy).

## Tests

```bash
pytest            # Flask inference service suite
```

The suite covers the service metadata, health check, request validation and
(when the model file is present) a full inference round-trip.

## Training your own model

`train.py` converts a labelled dataset to ELA, splits it into train/validation,
trains the fusion network and writes the model plus training curves.

```bash
python train.py \
  --data-dir data/CASIA2 \        # folder containing Au/ and Tp/ (or Real/ and Doctored/)
  --epochs 20 \
  --batch-size 32 \
  --model-out image_real_or_fake.keras
```

Or pass the folders explicitly:

```bash
python train.py --real-dir data/Au --fake-dir data/Tp --epochs 20
```

Useful flags: `--val-split`, `--image-size`, `--ela-quality`, `--learning-rate`,
`--limit` (quick smoke tests), `--no-pretrained` (offline), `--clean`.

## Docker

Run the whole stack with Docker Compose (builds all three images):

```bash
docker compose up --build
```

Then open <http://localhost:8080>. nginx serves the React build and proxies
`/api`, `/media` and `/admin` to the Django container.

## Notes on the model

The network was trained on the CASIA2 dataset. Two details matter for correct
inference:

1. **Input scale.** The shipped model was trained on **unscaled** (0–255) ELA
   pixels, so `model_utils.preprocess_image` uses `normalize=False`. Feeding
   0–1 inputs makes the output near-random.
2. **Output meaning.** The sigmoid is `P(real)` (see above). Reading it backwards
   flags authentic photos as forged.

The decision threshold defaults to `0.45` and can be tuned with
`CLASSIFICATION_THRESHOLD`.
