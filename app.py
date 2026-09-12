import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel
import uvicorn
import xgboost as xgb

app = FastAPI(title="PumpRadarAI", version="1.0")

# Перевіряємо, чи ми на Railway з підключеним Volume (/data)
BASE_DIR = Path("/data") if Path("/data").exists() else Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pump_radar_model.json"
DATASET_PATH = BASE_DIR / "pump_history.json"


class TokenMetric(BaseModel):
    ticker: str
    mcap: float
    volume_5m: float
    buys_5m: int
    sells_5m: int
    top10_pct: float
    is_pump: Optional[int] = None


def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    if MODEL_PATH.exists():
        model.load_model(MODEL_PATH)
    return model


def get_history_data() -> list:
    if DATASET_PATH.exists():
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history_data(data: list):
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.get("/")
def root():
    return {
        "status": "online",
        "has_model": MODEL_PATH.exists(),
        "history_records": len(get_history_data()),
        "storage_path": str(BASE_DIR),
    }


@app.post("/scan")
def scan_tokens(tokens: List[TokenMetric]):
    if not tokens:
        raise HTTPException(status_code=400, detail="Список токенів порожній")

    df = pd.DataFrame([t.model_dump() for t in tokens])
    features = df[["mcap", "volume_5m", "buys_5m", "sells_5m", "top10_pct"]]

    model = load_model()
    probs = model.predict_proba(features)[:, 1] if MODEL_PATH.exists() else np.zeros(len(df))

    results = []
    for idx, row in df.iterrows():
        prob = float(probs[idx])
        results.append({
            "ticker": row["ticker"],
            "mcap": row["mcap"],
            "pump_probability": round(prob, 4),
            "signal": "HIGH_PUMP_RISK" if prob >= 0.75 else "NORMAL",
        })

    return {"scan_results": results}


@app.post("/add_history")
def add_to_history(tokens: List[TokenMetric]):
    history = get_history_data()
    new_records = [t.model_dump() for t in tokens if t.is_pump is not None]

    if not new_records:
        raise HTTPException(status_code=400, detail="Не вказано is_pump (1 або 0)")

    history.extend(new_records)
    save_history_data(history)
    return {"message": f"Додано {len(new_records)} записів", "total_history": len(history)}


# --- Ендпоінти бэкапу / перенесення ---


@app.get("/export/model")
def export_model():
    if MODEL_PATH.exists():
        return FileResponse(MODEL_PATH, filename="pump_radar_model.json")
    raise HTTPException(status_code=404, detail="Файл моделі відсутній")


@app.get("/export/history")
def export_history():
    if DATASET_PATH.exists():
        return FileResponse(DATASET_PATH, filename="pump_history.json")
    raise HTTPException(status_code=404, detail="Файл історії відсутній")


@app.post("/import/model")
async def import_model(file: UploadFile = File(...)):
    content = await file.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(content)
    return {"status": "Модель успішно збережено на сервері"}


@app.post("/import/history")
async def import_history(file: UploadFile = File(...)):
    content = await file.read()
    with open(DATASET_PATH, "wb") as f:
        f.write(content)
    return {"status": "Історію успішно збережено на сервері"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)