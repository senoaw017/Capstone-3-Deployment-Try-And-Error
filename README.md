# Bike Sharing Predictor (Streamlit Cloud Ready)

## Deploy Steps
1. Place **all files at repo root** (this folder's contents): `app.py`, `utils.py`, `requirements.txt`, `runtime.txt`, and (optionally) `model.joblib` if < 95MB.
2. Push to GitHub → Streamlit Cloud → New app → pick repo → `app.py`.
3. If build fails, check **Logs → Build**. Most common fix is pinning NumPy/SciPy/Sklearn versions (already done here).

Pinned stack:
- Python 3.11 (via `runtime.txt`)
- numpy==1.26.4, scipy==1.11.4, scikit-learn==1.4.2
- streamlit>=1.36, pandas>=2.2, joblib>=1.3,<2, plotly>=5.20
