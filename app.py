
import io
import datetime as dt
import pandas as pd
import streamlit as st

from utils import (
    BikerSharingPreprocessor,
    load_model_and_predict,
)

st.set_page_config(page_title="Bike Sharing Predictor", page_icon="🚲", layout="wide")
st.title("🚲 Bike Sharing Predictor")

with st.sidebar:
    model_path = st.text_input("Model path", value="model.joblib")

col_date, col_time = st.columns(2)
with col_date:
    date_val = st.date_input(
        "Tanggal",
        value=dt.date.today(),
        min_value=dt.date(2000, 1, 1),
        max_value=dt.date(2100, 12, 31),
    )
with col_time:
    time_val = st.time_input("Jam", value=dt.time(8, 0))

col_t, col_h = st.columns(2)
with col_t:
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=22.5, step=0.5)
with col_h:
    hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.5)

WEATHERSIT_MAP = {
    "1 · Cerah/Terang": 1,
    "2 · Berawan/Mendung": 2,
    "3 · Hujan ringan / Salju ringan": 3,
    "4 · Hujan lebat / Salju": 4,
}
weathersit_label = st.selectbox("Kondisi Cuaca", list(WEATHERSIT_MAP.keys()), index=0)
weathersit = WEATHERSIT_MAP[weathersit_label]

SEASON_MAP = {
    "1 · Spring": 1,
    "2 · Summer": 2,
    "3 · Fall": 3,
    "4 · Winter": 4,
}
season_label = st.selectbox("Musim", list(SEASON_MAP.keys()), index=2)
season = SEASON_MAP[season_label]

is_holiday = st.checkbox("Hari Libur?", value=False)
holiday = 1 if is_holiday else 0

year = int(date_val.year)
month = int(date_val.month)
dayofweek = int(pd.Timestamp(date_val).weekday())
week = int(pd.Timestamp(date_val).isocalendar().week)
hr = int(time_val.hour)
is_weekend = 1 if dayofweek >= 5 else 0

inputs = {
    "hum": hum,
    "temp": temp,
    "hr": hr,
    "season": season,
    "month": month,
    "dayofweek": dayofweek,
    "week": week,
    "year": year,
    "weathersit": weathersit,
    "holiday": holiday,
    "is_weekend": is_weekend,
}

if st.button("🚀 Predict Demand", use_container_width=True):
    pre = BikerSharingPreprocessor()
    try:
        result = load_model_and_predict(model_path, inputs, pre)
        y = result["prediction"]
        st.success("Prediksi berhasil.")
        st.metric("Demand (predicted)", f"{y:,.2f}")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

st.caption("v4.4 (Py3.10 + NumPy 2 stack)")
