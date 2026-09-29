# Diabetic Retinopathy Stage Detection

Computer Vision coursework project: **5-stage diabetic retinopathy grading** from fundus images using EfficientNetB0 transfer learning, hybrid medical preprocessing, Softmax + **CORAL ordinal** heads, Grad-CAM explainability, and a Django screening product.

## Quick start (uv)

```bash
cd diabetic-retinopathy-detection
uv sync
bash scripts/setup_kaggle.sh "/home/neon-cultivator/Downloads/kaggle (1).json"
uv run python scripts/download_dataset.py
```

### Train (prefer Kaggle GPU; local CPU is slow)

```bash
# Softmax baseline
uv run python scripts/train_softmax.py

# CORAL ordinal
uv run python scripts/train_coral.py

# Evaluate + export winner
uv run python scripts/evaluate_export.py
```

### Product (Django)

```bash
uv run python showcase/manage.py runserver 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/` for Home, Product, and About. The screen reads `models/best_model.keras`.

### Docker

Weight files are not all stored in git. `models/best_model.keras` and `models/meta.json` must be present before the image is built.

```bash
docker compose up --build
```

The container serves gunicorn on port 8000 with `DJANGO_DEBUG=0`. Set `DJANGO_SECRET_KEY` in the environment before a public host.

### Streamlit (optional)

```bash
uv run streamlit run app/streamlit_app.py
```

### Notebooks (recommended learning path)

Open the **one complete step-by-step walkthrough** (all code inline, no `%run` scripts):

```bash
uv run jupyter lab notebooks/DR_Stage_Detection_Complete_Walkthrough.ipynb
```

Run cells top → bottom. The final **Export section** writes `models/best_model.keras` + `meta.json` for Streamlit.

Older split notebooks (`01_`…`04_`) remain for reference; CLI scripts under `scripts/` are unchanged.


## Documentation (mandatory evidence trail)

| Doc | Content |
|-----|---------|
| [docs/RESEARCH.md](docs/RESEARCH.md) | Problem, dataset comparison, novelty |
| [docs/DECISION_LOG.md](docs/DECISION_LOG.md) | Why each design choice |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System / model diagrams |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Every training run |
| [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md) | ≤20-page report skeleton |
| [docs/figures/](docs/figures/) | All plots and diagrams |

## Dataset

Primary: [`sachinkumar413/diabetic-retinopathy-dataset`](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset) — 2,750 fundus images, five ICDR stages. Locked split, seed 42: 1,924 train / 413 validation / 413 test.

## Disclaimer

Screening assistant only — **not** a medical diagnosis device.
