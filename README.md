# ALeaf-STENet Streamlit App

Hybrid deep learning dashboard for apple leaf disease classification, confidence
scoring, and disease severity percentage assessment using the trained
`aleaf_best.pth` checkpoint.

## Run Locally

Prerequisites: Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app expects `aleaf_best.pth` beside `app.py`. To use another location:

```bash
set ALEAF_MODEL_PATH=C:\path\to\aleaf_best.pth
streamlit run app.py
```

## Streamlit Deployment

Deploy with:

- `app.py` as the main file
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- the trained checkpoint `aleaf_best.pth`

For Streamlit Community Cloud, keep the large checkpoint in Git LFS or provide
it through the deployment environment and set `ALEAF_MODEL_PATH`.

## Model Output

The app loads the checkpoint classes directly:

- Alternaria leaf spot
- Brown spot
- Frogeye leaf spot
- Grey spot
- Health
- Mosaic
- Powdery mildew
- Rust
- Scab

Each upload returns the predicted class, confidence percentage, disease severity
percentage, top probabilities, and treatment guidance.
