from datetime import datetime, timedelta
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

app = FastAPI(title="PumpRadarAI", version="3.1")

BASE_DIR = Path("/data") if Path("/data").exists() else Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pump_radar_model.json"
DATASET_PATH = BASE_DIR / "pump_history.json"
TRACKING_PATH = BASE_DIR / "pending_tracks.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ВАШ_ТЕЛЕГРАМ_ТОКЕН")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "ВАШ_CHAT_ID")

def send_telegram_alert(message: str):
    if TELEGRAM_BOT_TOKEN == "ВАШ_ТЕЛЕГРАМ_ТОКЕН":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[TG ERROR] {e}")

def load_json(path: Path) -> list:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_json(path: Path, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42
    )
    if MODEL_PATH.exists():
        model.load_model(MODEL_PATH)
    return model

def track_peak_and_finalize():
    pending = load_json(TRACKING_PATH)
    if not pending:
        return

    now = datetime.now()
    still_pending = []
    history = load_json(DATASET_PATH)

    for item in pending:
        start_time = datetime.fromisoformat(item["start_time"])
        addr = item["address"]
        
        try:
            res = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{addr}", timeout=10)
            if res.status_code == 200:
                pairs = res.json().get("pairs", [])
                if pairs:
                    current_price = float(pairs[0].get("priceUsd", 0))
                    
                    if current_price > item["max_price"]:
                        item["max_price"] = current_price
                    
                    if now - start_time >= timedelta(minutes=30):
                        entry = item["entry_price"]
                        max_p = item["max_price"]
                        
                        max_gain_pct = float(((max_p - entry) / entry * 100) if entry > 0 else 0.0)
                        final_gain_pct = float(((current_price - entry) / entry * 100) if entry > 0 else 0.0)
                        
                        is_pump = 1 if max_gain_pct >= 50.0 else 0
                        status_emoji = "🚀 [УСПІШНИЙ ПАМП]" if is_pump else "❌ [НЕ ПАМПНУВСЯ]"
                        
                        result_msg = (
                            f"📊 <b>ЗВІТ СИГНАЛУ (30m Peak)</b>\n\n"
                            f"<b>Токен:</b> {item['ticker']}\n"
                            f"<b>Вхідний Score:</b> {item['score']:.1f}%\n"
                            f"<b>🔥 Макс. Ріст (ATH):</b> +{max_gain_pct:.1f}%\n"
                            f"<b>Поточний стан:</b> {final_gain_pct:+.1f}%\n"
                            f"<b>Результат:</b> {status_emoji}\n\n"
                            f"🔗 <a href='{item['url']}'>Відкрити графік</a>"
                        )
                        send_telegram_alert(result_msg)
                        
                        record = item["metrics"]
                        record["is_pump"] = is_pump
                        history.append(record)
                    else:
                        still_pending.append(item)
            else:
                still_pending.append(item)
        except Exception as e:
            print(f"[TRACKING ERROR] {e}")
            still_pending.append(item)

    save_json(TRACKING_PATH, still_pending)
    save_json(DATASET_PATH, history)

def auto_scan_and_alert():
    try:
        print("\n[SCANNER] Перевірка ринку...")
        response = requests.get("https://api.dexscreener.com/token-boosts/top/v1", timeout=10)
        if response.status_code != 200:
            return
        
        tokens_data = response.json()[:10]
        addresses = [t.get("tokenAddress") for t in tokens_data if t.get("tokenAddress")]
        if not addresses:
            return
            
        pairs_res = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{','.join(addresses[:10])}", timeout=10)
        pairs_data = pairs_res.json().get("pairs", []) if pairs_res.status_code == 200 else []
        pairs_dict = {p.get("baseToken", {}).get("address"): p for p in pairs_data if p.get("baseToken")}

        parsed_tokens = []
        for t in tokens_data:
            addr = t.get("tokenAddress", "")
            pair = pairs_dict.get(addr, {})
            
            mcap = float(pair.get("marketCap") or pair.get("fdv") or 0.0)
            v5m = float(pair.get("volume", {}).get("m5") or 0.0)
            tx5m = pair.get("txns", {}).get("m5", {})
            buys = int(tx5m.get("buys", 0))
            sells = int(tx5m.get("sells", 0))
            price_usd = float(pair.get("priceUsd", 0))
            
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
                "price": price_usd,
                "mcap": mcap,
                "volume_5m": v5m,
                "buys_5m": buys,
                "sells_5m": sells,
                "top10_pct": 20.0
            })

        df = pd.DataFrame(parsed_tokens)
        features = df[["mcap", "volume_5m", "buys_5m", "sells_5m", "top10_pct"]]

        if MODEL_PATH.exists():
            model = load_model()
            probs = model.predict_proba(features)[:, 1]
            
            pending_tracks = load_json(TRACKING_PATH)
            tracked_addrs = {p["address"] for p in pending_tracks}

            for idx, prob in enumerate(probs):
                tok = parsed_tokens[idx]
                score_pct = float(prob * 100)
                
                if prob >= 0.75 and tok["address"] not in tracked_addrs:
                    print(f"🔥 [PUMP ALERT] {tok['ticker']} | Score: {score_pct:.1f}% | MCap: ${tok['mcap']:,.0f}")
                    
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

                    pending_tracks.append({
                        "ticker": str(tok["ticker"]),
                        "address": str(tok["address"]),
                        "url": str(tok["url"]),
                        "score": float(score_pct),
                        "entry_price": float(tok["price"]),
                        "max_price": float(tok["price"]),
                        "start_time": datetime.now().isoformat(),
                        "metrics": {
                            "mcap": float(tok["mcap"]),
                            "volume_5m": float(tok["volume_5m"]),
                            "buys_5m": int(tok["buys_5m"]),
                            "sells_5m": int(tok["sells_5m"]),
                            "top10_pct": float(tok["top10_pct"])
                        }
                    })
                    save_json(TRACKING_PATH, pending_tracks)
                else:
                    print(f"[SCAN] {tok['ticker']} | Score: {score_pct:.1f}% | MCap: ${tok['mcap']:,.0f}")

    except Exception as e:
        print(f"[SCAN ERROR] {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(auto_scan_and_alert, 'interval', seconds=60)
scheduler.add_job(track_peak_and_finalize, 'interval', seconds=60)
scheduler.start()

@app.get("/")
def root():
    return {
        "status": "online",
        "has_model": MODEL_PATH.exists(),
        "history_records": len(load_json(DATASET_PATH)),
        "active_tracks": len(load_json(TRACKING_PATH)),
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