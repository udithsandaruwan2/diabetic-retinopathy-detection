# Diabetic Retinopathy Stage Detection

Computer Vision coursework project: **5-stage diabetic retinopathy grading** from fundus images using EfficientNetB0 transfer learning, hybrid medical preprocessing, Softmax + **CORAL ordinal** heads, Grad-CAM explainability, and a Streamlit real-time demo.

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

### Demo

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

Primary: [`sovitrath/diabetic-retinopathy-224x224-2019-data`](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) (APTOS 2019, 5 classes).

## Disclaimer

Screening assistant only — **not** a medical diagnosis device.
