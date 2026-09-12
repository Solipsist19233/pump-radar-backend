import json
import os
from pathlib import Path
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel
import requests
import uvicorn
import xgboost as xgb

app = FastAPI(title="PumpRadarAI", version="2.0")

# Перевірка на постійний диск Railway (/data)
BASE_DIR = Path("/data") if Path("/data").exists() else Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pump_radar_model.json"
DATASET_PATH = BASE_DIR / "pump_history.json"

# Налаштування Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ВАШ_ТЕЛЕГРАМ_ТОКЕН")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "ВАШ_CHAT_ID")

def send_telegram_alert(message: str):
    """Надсилання повідомлення у Telegram"""
    if TELEGRAM_BOT_TOKEN == "ВАШ_ТЕЛЕГРАМ_ТОКЕН":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
        print("[TG] Сповіщення успішно надіслано в Telegram")
    except Exception as e:
        print(f"[TG ERROR] {e}")

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
        n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42
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

def auto_scan_and_alert():
    """Автоматична функція фонового сканування DEX з посиланнями та адресами"""
    try:
        print("\n[SCANNER] Початок перевірки ринку...")
        
        response = requests.get("https://api.dexscreener.com/token-boosts/top/v1", timeout=10)
        if response.status_code != 200:
            print(f"[SCANNER] Помилка API DexScreener: {response.status_code}")
            return
        
        tokens_data = response.json()[:10]
        parsed_tokens = []
        
        addresses = [t.get("tokenAddress") for t in tokens_data if t.get("tokenAddress")]
        if not addresses:
            print("[SCANNER] Не знайдено адрес токенів.")
            return
            
        pairs_res = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{','.join(addresses[:10])}", timeout=10)
        pairs_data = pairs_res.json().get("pairs", []) if pairs_res.status_code == 200 else []
        
        pairs_dict = {p.get("baseToken", {}).get("address"): p for p in pairs_data if p.get("baseToken")}

        for t in tokens_data:
            addr = t.get("tokenAddress", "")
            pair = pairs_dict.get(addr, {})
            
            mcap = float(pair.get("marketCap") or pair.get("fdv") or 0.0)
            v5m = float(pair.get("volume", {}).get("m5") or 0.0)
            
            tx5m = pair.get("txns", {}).get("m5", {})
            buys = int(tx5m.get("buys", 0))
            sells = int(tx5m.get("sells", 0))
            
            if mcap == 0:
                mcap = float(pair.get("liquidity", {}).get("usd", 0) * 2 or 10000.0)
            if v5m == 0:
                v5m = float(pair.get("volume", {}).get("h1", 0) / 12)

            symbol = pair.get("baseToken", {}).get("symbol") or addr[:8]
            pair_url = pair.get("url") or f"https://dexscreener.com/search?q={addr}"

            parsed_tokens.append({
                "ticker": symbol,
                "address": addr,
                "url": pair_url,
                "mcap": mcap,
                "volume_5m": v5m,
                "buys_5m": buys,
                "sells_5m": sells,
                "top10_pct": 20.0
            })
            
        if not parsed_tokens:
            print("[SCANNER] Токенів для аналізу не знайдено.")
            return

        print(f"[SCANNER] Знайдено {len(parsed_tokens)} токенів. Оцінюю через XGBoost...")

        df = pd.DataFrame(parsed_tokens)
        features = df[["mcap", "volume_5m", "buys_5m", "sells_5m", "top10_pct"]]

        if MODEL_PATH.exists():
            model = load_model()
            probs = model.predict_proba(features)[:, 1]
            
            for idx, prob in enumerate(probs):
                tok = parsed_tokens[idx]
                score_pct = prob * 100
                
                if prob >= 0.75:
                    print(f"🔥 [PUMP DETECTED] {tok['ticker']} | Score: {score_pct:.1f}% | MCap: ${tok['mcap']:,.0f}")
                    msg = (
                        f"🚀 <b>PUMP ALERT!</b>\n\n"
                        f"<b>Токен:</b> {tok['ticker']}\n"
                        f"<b>Ймовірність пампа:</b> {score_pct:.1f}%\n"
                        f"<b>MCap:</b> ${tok['mcap']:,.0f}\n"
                        f"<b>Об'єм 5хв:</b> ${tok['volume_5m']:,.0f}\n\n"
                        f"<b>Адреса:</b> <code>{tok['address']}</code>\n"
                        f"🔗 <a href='{tok['url']}'>Відкрити на DexScreener</a>"
                    )
                    send_telegram_alert(msg)
                else:
                    print(f"[SKIP] {tok['ticker']} | Score: {score_pct:.1f}% | MCap: ${tok['mcap']:,.0f}")
        else:
            print("[SCANNER] Файл моделі не знайдено на диску.")

    except Exception as e:
        print(f"[SCAN ERROR] {e}")

# Фоновий планувальник (запуск сканера кожні 60 секунд)
scheduler = BackgroundScheduler()
scheduler.add_job(auto_scan_and_alert, 'interval', seconds=60)
scheduler.start()

@app.get("/")
def root():
    return {
        "status": "online",
        "has_model": MODEL_PATH.exists(),
        "history_records": len(get_history_data()),
        "telegram_configured": TELEGRAM_BOT_TOKEN != "ВАШ_ТЕЛЕГРАМ_ТОКЕН"
    }

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
    return {"status": "Модель успішно імпортовано"}

@app.post("/import/history")
async def import_history(file: UploadFile = File(...)):
    content = await file.read()
    with open(DATASET_PATH, "wb") as f:
        f.write(content)
    return {"status": "Історію успішно імпортовано"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)