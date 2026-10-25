from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR_ENV = os.getenv("MODEL_DIR")
MODEL_DIR = (
    Path(MODEL_DIR_ENV)
    if MODEL_DIR_ENV and Path(MODEL_DIR_ENV).is_absolute()
    else BASE_DIR / (MODEL_DIR_ENV or "checkpoints/HanoiWeather_SOFTS_FINAL_168_24")
)
WEIGHTS_PATH = MODEL_DIR / "checkpoint.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE = float(os.getenv("WEATHER_LATITUDE", "21.0278"))
LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "105.8342"))
TIMEZONE = "Asia/Ho_Chi_Minh"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]
FEATURE_LABELS = {
    "temperature_2m": {"label": "Nhiệt độ", "unit": "°C"},
    "relative_humidity_2m": {"label": "Độ ẩm", "unit": "%"},
    "precipitation": {"label": "Lượng mưa", "unit": "mm"},
    "wind_speed_10m": {"label": "Tốc độ gió", "unit": "km/h"},
    "cloud_cover": {"label": "Độ che phủ mây", "unit": "%"},
}

app = FastAPI(
    title="HanoiWeather AI API",
    version="1.0.0",
    description="SOFTS hourly weather forecasting API for Hanoi.",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_model = None
_scaler = None
_config: dict[str, Any] = {}
_model_error: str | None = None
_model_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _cached(key: str, ttl_seconds: int, producer):
    now = time.time()
    with _cache_lock:
        item = _cache.get(key)
        if item and item[0] > now:
            return item[1]
    value = producer()
    with _cache_lock:
        _cache[key] = (now + ttl_seconds, value)
    return value


def _load_model() -> None:
    """Load the exported SOFTS checkpoint once. The website still serves live provider data if absent."""
    global _model, _scaler, _config, _model_error

    missing = [str(path) for path in (WEIGHTS_PATH, SCALER_PATH, CONFIG_PATH) if not path.exists()]
    if missing:
        _model_error = "Thiếu tệp mô hình: " + ", ".join(Path(p).name for p in missing)
        return

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

        features = config.get("features", DEFAULT_FEATURES)
        if features != DEFAULT_FEATURES:
            raise ValueError(
                "Danh sách features trong config.json không khớp với API. "
                f"Nhận được: {features}"
            )

        # Use the same architecture settings as the final SOFTS experiment.
        args = SimpleNamespace(
            model="SOFTS",
            seq_len=int(config.get("seq_len", 168)),
            pred_len=int(config.get("pred_len", 24)),
            features="M",
            freq=config.get("freq", "h"),
            d_model=int(config.get("d_model", 128)),
            d_core=int(config.get("d_core", 64)),
            d_ff=int(config.get("d_ff", 128)),
            e_layers=int(config.get("e_layers", 2)),
            dropout=float(config.get("dropout", 0.0)),
            activation=config.get("activation", "gelu"),
            use_norm=bool(config.get("use_norm", True)),
        )

        from models import SOFTS
        model = SOFTS.Model(args).float().to(DEVICE)
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()

        scaler = joblib.load(SCALER_PATH)
        if getattr(scaler, "n_features_in_", len(features)) != len(features):
            raise ValueError("scaler.pkl không có đúng 5 biến đầu vào.")

        _model = model
        _scaler = scaler
        _config = config
        _model_error = None
    except Exception as exc:  # report diagnostics through /api/status
        _model = None
        _scaler = None
        _config = {}
        _model_error = f"Không thể tải mô hình: {type(exc).__name__}: {exc}"


@app.on_event("startup")
def startup_event() -> None:
    _load_model()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "HanoiWeather AI",
        "model_loaded": _model is not None,
        "model_name": "SOFTS",
        "device": str(DEVICE),
        "model_error": _model_error,
    }


@app.get("/api/status")
def status():
    return {
        "model_loaded": _model is not None,
        "model_name": "SOFTS",
        "model_artifacts": {
            "checkpoint": WEIGHTS_PATH.exists(),
            "scaler": SCALER_PATH.exists(),
            "config": CONFIG_PATH.exists(),
        },
        "model_error": _model_error,
        "training_data_end": _config.get("data_end", "2025-12-31"),
        "device": str(DEVICE),
        "weather_source": "Open-Meteo Forecast API",
        "timezone": TIMEZONE,
    }


def _open_meteo(params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Không lấy được dữ liệu từ Open-Meteo: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Open-Meteo trả về JSON không hợp lệ.") from exc
    if "error" in payload and payload.get("error"):
        raise HTTPException(status_code=502, detail=payload.get("reason", "Open-Meteo API error"))
    return payload


def _base_params() -> dict[str, Any]:
    return {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


@app.get("/api/weather/current")
def current_weather():
    def produce():
        params = _base_params()
        params["current"] = ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "wind_speed_10m", "cloud_cover", "weather_code", "is_day",
        ])
        params["forecast_days"] = 1
        payload = _open_meteo(params)
        current = payload.get("current")
        if not current:
            raise HTTPException(status_code=502, detail="API không trả về current weather.")
        return {
            "location": "Hà Nội, Việt Nam",
            "latitude": payload.get("latitude", LATITUDE),
            "longitude": payload.get("longitude", LONGITUDE),
            "timezone": payload.get("timezone", TIMEZONE),
            "time": current.get("time"),
            "temperature_2m": current.get("temperature_2m"),
            "relative_humidity_2m": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed_10m": current.get("wind_speed_10m"),
            "cloud_cover": current.get("cloud_cover"),
            "weather_code": current.get("weather_code"),
            "is_day": current.get("is_day"),
            "source": "Open-Meteo Forecast API",
            "note": "Điều kiện hiện tại do Open-Meteo cung cấp; không phải đầu ra SOFTS.",
        }
    return _cached("current", 600, produce)


@app.get("/api/weather/daily")
def daily_weather():
    def produce():
        params = _base_params()
        params["daily"] = ",".join([
            "weather_code", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "precipitation_probability_max", "wind_speed_10m_max",
        ])
        params["forecast_days"] = 5
        payload = _open_meteo(params)
        daily = payload.get("daily", {})
        times = daily.get("time", [])
        items = []
        for i, day in enumerate(times):
            def at(name, default=None):
                values = daily.get(name, [])
                return values[i] if i < len(values) else default
            items.append({
                "date": day,
                "weather_code": at("weather_code"),
                "temperature_max": at("temperature_2m_max"),
                "temperature_min": at("temperature_2m_min"),
                "precipitation_sum": at("precipitation_sum"),
                "precipitation_probability_max": at("precipitation_probability_max"),
                "wind_speed_max": at("wind_speed_10m_max"),
            })
        return {
            "location": "Hà Nội, Việt Nam",
            "timezone": payload.get("timezone", TIMEZONE),
            "daily": items,
            "source": "Open-Meteo Forecast API",
            "note": "Dự báo nhiều ngày do Open-Meteo cung cấp, không phải SOFTS.",
        }
    return _cached("daily", 900, produce)


def _weather_code_text(code: int | None) -> str:
    if code is None:
        return "Chưa rõ"
    if code == 0:
        return "Trời quang"
    if code in (1, 2):
        return "Ít mây"
    if code == 3:
        return "Nhiều mây"
    if code in (45, 48):
        return "Sương mù"
    if code in (51, 53, 55, 56, 57):
        return "Mưa phùn"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "Có mưa"
    if code in (71, 73, 75, 77, 85, 86):
        return "Tuyết"
    if code in (95, 96, 99):
        return "Dông"
    return "Biến đổi thời tiết"


@app.get("/api/forecast/ai")
def ai_forecast():
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Mô hình SOFTS chưa sẵn sàng. Hãy đặt checkpoint.pth, scaler.pkl "
                "và config.json vào checkpoints/HanoiWeather_SOFTS_FINAL_168_24/ "
                "rồi khởi động lại dịch vụ. " + (_model_error or "")
            ),
        )

    def produce():
        # Pull extra past hours so that a provider edge-case cannot leave fewer than 168 rows.
        params = _base_params()
        params["hourly"] = ",".join(DEFAULT_FEATURES)
        params["past_hours"] = 192
        params["forecast_hours"] = 24
        payload = _open_meteo(params)
        hourly = payload.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), errors="coerce")
        if len(times) == 0:
            raise HTTPException(status_code=502, detail="Nguồn thời tiết không trả về chuỗi theo giờ.")

        frame = pd.DataFrame({"date": times})
        for feature in DEFAULT_FEATURES:
            values = hourly.get(feature)
            if values is None:
                raise HTTPException(status_code=502, detail=f"Nguồn dữ liệu thiếu biến {feature}.")
            frame[feature] = values

        now_local = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None, minute=0, second=0, microsecond=0)
        history = frame.loc[frame["date"] <= pd.Timestamp(now_local)].dropna(subset=DEFAULT_FEATURES)
        history = history.sort_values("date").drop_duplicates("date").tail(168).reset_index(drop=True)
        if len(history) < 168:
            raise HTTPException(
                status_code=502,
                detail=f"Chỉ nhận được {len(history)} giờ dữ liệu; cần đúng 168 giờ liên tục.",
            )
        expected = pd.date_range(end=history["date"].iloc[-1], periods=168, freq="h")
        if not pd.DatetimeIndex(history["date"]).equals(expected):
            raise HTTPException(status_code=502, detail="Dữ liệu đầu vào không liên tục theo giờ; không thể dự báo an toàn.")

        raw_values = history[DEFAULT_FEATURES].to_numpy(dtype=np.float64)
        scaled = _scaler.transform(raw_values).astype(np.float32)
        # Match the time-marker transform used by the SOFTS training Dataset_Custom.
        from utils.timefeatures import time_features
        time_marks = time_features(pd.to_datetime(history["date"].values), freq="h").transpose(1, 0)

        x = torch.from_numpy(scaled).unsqueeze(0).to(DEVICE)
        x_mark = torch.as_tensor(time_marks, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with _model_lock, torch.inference_mode():
            prediction_scaled = _model(x, x_mark, None, None).detach().cpu().numpy()[0]

        prediction = _scaler.inverse_transform(prediction_scaled).astype(float)
        pos = {name: idx for idx, name in enumerate(DEFAULT_FEATURES)}
        prediction[:, pos["precipitation"]] = np.maximum(prediction[:, pos["precipitation"]], 0)
        prediction[:, pos["wind_speed_10m"]] = np.maximum(prediction[:, pos["wind_speed_10m"]], 0)
        prediction[:, pos["relative_humidity_2m"]] = np.clip(prediction[:, pos["relative_humidity_2m"]], 0, 100)
        prediction[:, pos["cloud_cover"]] = np.clip(prediction[:, pos["cloud_cover"]], 0, 100)

        last_input_time = pd.Timestamp(history["date"].iloc[-1])
        forecast_times = [
            (last_input_time + pd.Timedelta(hours=i)).isoformat(timespec="minutes")
            for i in range(1, 25)
        ]
        forecast_rows = []
        for i, forecast_time in enumerate(forecast_times):
            row = {"time": forecast_time}
            for j, feature in enumerate(DEFAULT_FEATURES):
                row[feature] = round(float(prediction[i, j]), 4)
            forecast_rows.append(row)

        return {
            "location": "Hà Nội, Việt Nam",
            "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
            "last_input_time": last_input_time.isoformat(timespec="minutes"),
            "input_hours": 168,
            "forecast_hours": 24,
            "forecast": forecast_rows,
            "source": "Open-Meteo hourly model data (past 168 hours) + SOFTS",
            "training_data_end": _config.get("data_end", "2025-12-31"),
            "note": (
                "Đây là dự báo thử nghiệm của SOFTS. Mô hình được huấn luyện trên ERA5 đến cuối năm 2025; "
                "đầu vào gần hiện tại đến từ Open-Meteo, nên khác nguồn và có thể có sai lệch miền dữ liệu. "
                "Không dùng thay thế cảnh báo khí tượng chính thức."
            ),
        }
    return _cached("ai_forecast", 600, produce)


@app.get("/api/forecast/metrics")
def model_metrics():
    return {
        "model": "SOFTS",
        "evaluation": "historical test split",
        "metrics": [
            {"feature": "temperature_2m", "label": "Nhiệt độ", "unit": "°C", "mae": 1.3154, "rmse": 1.8124},
            {"feature": "relative_humidity_2m", "label": "Độ ẩm", "unit": "%", "mae": 5.5028, "rmse": 7.4068},
            {"feature": "precipitation", "label": "Lượng mưa", "unit": "mm", "mae": 0.2441, "rmse": 0.7633},
            {"feature": "wind_speed_10m", "label": "Tốc độ gió", "unit": "km/h", "mae": 2.3621, "rmse": 3.1688},
            {"feature": "cloud_cover", "label": "Độ che phủ mây", "unit": "%", "mae": 18.2652, "rmse": 25.6142},
        ],
        "dino_note": "Thí nghiệm encoder DINOv3 có MAE cao hơn SOFTS gốc ở cả 5 biến trong cấu hình đã thử; xem kết quả nghiên cứu trong repository.",
        "warning": "Các chỉ số là backtest lịch sử, không phải độ chính xác được bảo đảm cho dự báo trực tiếp.",
    }


@app.get("/api/weather/code/{code}")
def weather_code(code: int):
    return {"code": code, "description": _weather_code_text(code)}
