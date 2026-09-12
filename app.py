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

app = FastAPI(title="PumpRadarAI - Early Accumulation Engine", version="4.0")

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
        n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42
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
                            f"📊 <b>ЗВІТ РАННЬОГО СИГНАЛУ (30m)</b>\n\n"
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

def fetch_target_tokens() -> list:
    addresses = set()
    try:
        r1 = requests.get("https://api.dexscreener.com/token-boosts/top/v1", timeout=5)
        if r1.status_code == 200:
            for item in r1.json()[:30]:
                if item.get("tokenAddress"):
                    addresses.add(item.get("tokenAddress"))
    except Exception as e:
        print(f"[FETCH ERROR 1] {e}")

    try:
        r2 = requests.get("https://api.dexscreener.com/token-profiles/latest/v1", timeout=5)
        if r2.status_code == 200:
            for item in r2.json()[:30]:
                if item.get("tokenAddress"):
                    addresses.add(item.get("tokenAddress"))
    except Exception as e:
        print(f"[FETCH ERROR 2] {e}")

    return list(addresses)[:50]

def auto_scan_and_alert():
    try:
        print("\n[SCANNER] Перевірка монет на тихий закуп (ранній старт)...")
        addresses = fetch_target_tokens()
        if not addresses:
            return

        chunk_size = 30
        pairs_data = []
        for i in range(0, len(addresses), chunk_size):
            chunk = addresses[i:i + chunk_size]
            pairs_res = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{','.join(chunk)}", timeout=10)
            if pairs_res.status_code == 200:
                pairs_data.extend(pairs_res.json().get("pairs", []) or [])

        pairs_dict = {}
        for p in pairs_data:
            base_addr = p.get("baseToken", {}).get("address")
            if base_addr and base_addr not in pairs_dict:
                pairs_dict[base_addr] = p

        parsed_tokens = []
        for addr in addresses:
            pair = pairs_dict.get(addr, {})
            if not pair:
                continue
            
            mcap = float(pair.get("marketCap") or pair.get("fdv") or 0.0)
            v5m = float(pair.get("volume", {}).get("m5") or 0.0)
            price_change_5m = float(pair.get("priceChange", {}).get("m5") or 0.0)
            
            tx5m = pair.get("txns", {}).get("m5", {})
            buys = int(tx5m.get("buys", 0))
            sells = int(tx5m.get("sells", 0))
            price_usd = float(pair.get("priceUsd", 0))
            
            # ФІЛЬТР 1: Беремо тільки дно ($3k - $40k MCap)
            if not (3000 <= mcap <= 40000):
                continue

            # ФІЛЬТР 2: Ціна ще НЕ повинна була вистрілити (від -10% до +25%)
            if price_change_5m > 25.0:
                continue

            buy_ratio = float(buys / (sells + 1))
            vol_mcap_ratio = float(v5m / (mcap + 1))

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
                "price_change_5m": price_change_5m,
                "buy_ratio": buy_ratio,
                "vol_mcap_ratio": vol_mcap_ratio
            })

        if not parsed_tokens:
            return

        df = pd.DataFrame(parsed_tokens)
        feature_cols = ["mcap", "volume_5m", "buys_5m", "sells_5m", "price_change_5m", "buy_ratio", "vol_mcap_ratio"]
        features = df[feature_cols]

        if MODEL_PATH.exists():
            model = load_model()
            probs = model.predict_proba(features)[:, 1]
            
            pending_tracks = load_json(TRACKING_PATH)
            tracked_addrs = {p["address"] for p in pending_tracks}

            for idx, prob in enumerate(probs):
                tok = parsed_tokens[idx]
                score_pct = float(prob * 100)
                
                # Порог спрацювання 70% на ранньому накопиченні
                if prob >= 0.70 and tok["address"] not in tracked_addrs:
                    print(f"💎 [EARLY ACCUMULATION ALERT] {tok['ticker']} | Score: {score_pct:.1f}% | MCap: ${tok['mcap']:,.0f}")
                    
                    msg = (
                        f"💎 <b>РAННІЙ СИГНАЛ (НАКОПИЧЕННЯ)</b>\n\n"
                        f"<b>Токен:</b> {tok['ticker']}\n"
                        f"<b>Score моделі:</b> {score_pct:.1f}%\n"
                        f"<b>MCap:</b> ${tok['mcap']:,.0f}\n"
                        f"<b>Зміна ціни 5m:</b> {tok['price_change_5m']:+.1f}%\n"
                        f"<b>Buy/Sell Ratio:</b> {tok['buy_ratio']:.1f}x\n"
                        f"<b>Об'єм 5m:</b> ${tok['volume_5m']:,.0f}\n\n"
                        f"<b>Адреса:</b> <code>{tok['address']}</code>\n"
                        f"🔗 <a href='{tok['url']}'>Відкрити графік</a>"
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
                            "price_change_5m": float(tok["price_change_5m"]),
                            "buy_ratio": float(tok["buy_ratio"]),
                            "vol_mcap_ratio": float(tok["vol_mcap_ratio"])
                        }
                    })
                    save_json(TRACKING_PATH, pending_tracks)
                else:
                    print(f"[SCAN] {tok['ticker']} | Score: {score_pct:.1f}% | 5m Change: {tok['price_change_5m']:+.1f}%")

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

# ЕНДПОІНТ ДЛЯ ПОВНОГО СКАСУВАННЯ ТА ЗAЧИСТКИ СТАРОЇ ІСТОРІЇ
@app.post("/reset-history")
def reset_history():
    save_json(DATASET_PATH, [])
    save_json(TRACKING_PATH, [])
    if MODEL_PATH.exists():
        os.remove(MODEL_PATH)
    return {"status": "success", "message": "Історію та стару модель повністю вилучено. Готово до нового збору з нуля!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)