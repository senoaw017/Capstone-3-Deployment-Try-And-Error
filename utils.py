
from typing import Dict, Any, List
import pandas as pd
import joblib
import os

FEATURES: List[str] = [
    "hum", "temp", "hr",
    "season", "month", "dayofweek", "week", "year",
    "weathersit", "holiday", "is_weekend"
]

class BikerSharingPreprocessor:
    def transform(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        row = {}
        for f in FEATURES:
            v = inputs.get(f, None)
            try:
                if f in ("hum", "temp"):
                    row[f] = float(v) if v is not None else None
                else:
                    row[f] = int(v) if v is not None else None
            except Exception:
                row[f] = None
        return pd.DataFrame([row], columns=FEATURES)

def _resolve_model_path(path_hint: str) -> str:
    candidates = [path_hint, "model.joblib", "bike_sharing_pipeline.joblib"]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return path_hint

def load_model_and_predict(model_path: str, inputs: Dict[str, Any], preprocessor: BikerSharingPreprocessor) -> Dict[str, Any]:
    model_path = _resolve_model_path(model_path)
    model = joblib.load(model_path)
    X = preprocessor.transform(inputs)
    yhat = model.predict(X)
    try:
        pred = float(yhat[0])
    except Exception:
        pred = float(yhat)
    return {"prediction": pred}
