from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
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

app = FastAPI(title="PumpRadarAI - Paper Trading Engine", version="5.2")

BASE_DIR = Path("/data") if Path("/data").exists() else Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent

# Перевіряємо модель і в /data, і в корені проєкту
MODEL_PATH = BASE_DIR / "pump_radar_model.json"
if not MODEL_PATH.exists() and (ROOT_DIR / "pump_radar_model.json").exists():
    MODEL_PATH = ROOT_DIR / "pump_radar_model.json"

DATASET_PATH = BASE_DIR / "pump_history.json"
TRACKING_PATH = BASE_DIR / "pending_tracks.json"
PORTFOLIO_PATH = BASE_DIR / "paper_portfolio.json"

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

def load_portfolio() -> dict:
    if PORTFOLIO_PATH.exists():
        try:
            with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"balance": 10000.0, "total_trades": 0, "wins": 0, "losses": 0, "total_realized_pnl": 0.0}

def save_portfolio(portfolio: dict):
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)

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
    portfolio = load_portfolio()

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

                    entry = item["entry_price"]
                    max_p = item["max_price"]
                    
                    max_gain_pct = float(((max_p - entry) / entry * 100) if entry > 0 else 0.0)
                    current_gain_pct = float(((current_price - entry) / entry * 100) if entry > 0 else 0.0)

                    if item.get("paper_trade"):
                        # Take 1: +100% (2x) -> Продаж 50% ($5) -> Повертаємо $10.0
                        if max_gain_pct >= 100.0 and not item["paper_trade"].get("take1_hit"):
                            item["paper_trade"]["take1_hit"] = True
                            portfolio["balance"] += 10.0
                            portfolio["total_realized_pnl"] += 5.0
                            save_portfolio(portfolio)
                            send_telegram_alert(
                                f"🎯 <b>PAPER TRADE: TAKE 1 HIT (2x / +100%)</b>\n\n"
                                f"<b>Токен:</b> {item['ticker']}\n"
                                f"<b>Дія:</b> Продано 50% ($5.0)\n"
                                f"<b>Повернуто в банк:</b> +$10.0\n"
                                f"💰 <b>Баланс:</b> ${portfolio['balance']:.2f}"
                            )

                        # Take 2: +800% (9x) -> Продаж 30% ($3) -> Забираємо $27.0
                        if max_gain_pct >= 800.0 and not item["paper_trade"].get("take2_hit"):
                            item["paper_trade"]["take2_hit"] = True
                            portfolio["balance"] += 27.0
                            portfolio["total_realized_pnl"] += 24.0
                            save_portfolio(portfolio)
                            send_telegram_alert(
                                f"🚀 <b>PAPER TRADE: TAKE 2 HIT (9x / +800%)</b>\n\n"
                                f"<b>Токен:</b> {item['ticker']}\n"
                                f"<b>Дія:</b> Продано 30% ($3.0)\n"
                                f"<b>Зараховано:</b> +$27.0\n"
                                f"💰 <b>Баланс:</b> ${portfolio['balance']:.2f}"
                            )

                        # Take 3: +1500% (16x) -> Продаж 20% ($2) -> Забираємо $32.0
                        if max_gain_pct >= 1500.0 and not item["paper_trade"].get("take3_hit"):
                            item["paper_trade"]["take3_hit"] = True
                            portfolio["balance"] += 32.0
                            portfolio["total_realized_pnl"] += 30.0
                            save_portfolio(portfolio)
                            send_telegram_alert(
                                f"🌕 <b>PAPER TRADE: TAKE 3 MOONBAG (16x / +1500%)</b>\n\n"
                                f"<b>Токен:</b> {item['ticker']}\n"
                                f"<b>Дія:</b> Фінал Moonbag 20% ($2.0)\n"
                                f"<b>Зараховано:</b> +$32.0\n"
                                f"💰 <b>Баланс:</b> ${portfolio['balance']:.2f}"
                            )

                    if now - start_time >= timedelta(minutes=30):
                        is_pump = 1 if max_gain_pct >= 50.0 else 0
                        status_emoji = "🚀 [УСПІШНИЙ ПАМП]" if is_pump else "❌ [НЕ ПАМПНУВСЯ]"
                        
                        if is_pump:
                            portfolio["wins"] += 1
                        else:
                            portfolio["losses"] += 1
                        portfolio["total_trades"] += 1

                        result_msg = (
                            f"📊 <b>ЗВІТ РАННЬОГО СИГНАЛУ (30m)</b>\n\n"
                            f"<b>Токен:</b> {item['ticker']}\n"
                            f"<b>Вхідний Score:</b> {item.get('score', 0):.1f}%\n"
                            f"<b>🔥 Макс. Ріст (ATH):</b> +{max_gain_pct:.1f}%\n"
                            f"<b>Поточний стан:</b> {current_gain_pct:+.1f}%\n"
                            f"<b>Результат:</b> {status_emoji}\n\n"
                            f"💼 <b>Paper Portfolio:</b> ${portfolio['balance']:.2f} | Wins: {portfolio['wins']}/{portfolio['total_trades']}\n"
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
    save_portfolio(portfolio)

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
            
            if not (3000 <= mcap <= 40000) or price_change_5m > 25.0:
                continue

            buy_ratio = float(buys / (sells + 1))
            vol_mcap_ratio = float(v5m / (mcap + 1))
            symbol = pair.get("baseToken", {}).get("symbol") or addr[:8]
            pair_url = pair.get("url") or f"https://dexscreener.com/search?q={addr}"

            parsed_tokens.append({
                "ticker": symbol, "address": addr, "url": pair_url, "price": price_usd,
                "mcap": mcap, "volume_5m": v5m, "buys_5m": buys, "sells_5m": sells,
                "price_change_5m": price_change_5m, "buy_ratio": buy_ratio, "vol_mcap_ratio": vol_mcap_ratio
            })

        if not parsed_tokens:
            return

        df = pd.DataFrame(parsed_tokens)
        feature_cols = ["mcap", "volume_5m", "buys_5m", "sells_5m", "price_change_5m", "buy_ratio", "vol_mcap_ratio"]
        features = df[feature_cols]

        pending_tracks = load_json(TRACKING_PATH)
        tracked_addrs = {p["address"] for p in pending_tracks}
        portfolio = load_portfolio()

        if MODEL_PATH.exists():
            model = load_model()
            probs = model.predict_proba(features)[:, 1]
            
            for idx, prob in enumerate(probs):
                tok = parsed_tokens[idx]
                score_pct = float(prob * 100)
                
                if prob >= 0.70 and tok["address"] not in tracked_addrs:
                    portfolio["balance"] -= 10.0
                    save_portfolio(portfolio)

                    msg = (
                        f"💎 <b>РAННІЙ СИГНАЛ (НАКОПИЧЕННЯ)</b>\n\n"
                        f"<b>Токен:</b> {tok['ticker']}\n"
                        f"<b>Score моделі:</b> {score_pct:.1f}%\n"
                        f"<b>MCap:</b> ${tok['mcap']:,.0f}\n"
                        f"<b>Buy/Sell Ratio:</b> {tok['buy_ratio']:.1f}x\n\n"
                        f"💵 <b>Paper Entry:</b> $10.0 ордер\n"
                        f"🔗 <a href='{tok['url']}'>Відкрити графік</a>"
                    )
                    send_telegram_alert(msg)

                    pending_tracks.append({
                        "ticker": str(tok["ticker"]), "address": str(tok["address"]), "url": str(tok["url"]),
                        "score": float(score_pct), "entry_price": float(tok["price"]), "max_price": float(tok["price"]),
                        "start_time": datetime.now().isoformat(),
                        "paper_trade": {
                            "position_usd": 10.0, 
                            "take1_hit": False, 
                            "take2_hit": False,
                            "take3_hit": False
                        },
                        "metrics": tok
                    })
                    save_json(TRACKING_PATH, pending_tracks)
        else:
            for tok in parsed_tokens:
                if tok["address"] not in tracked_addrs:
                    pending_tracks.append({
                        "ticker": str(tok["ticker"]), "address": str(tok["address"]), "url": str(tok["url"]),
                        "score": 0.0, "entry_price": float(tok["price"]), "max_price": float(tok["price"]),
                        "start_time": datetime.now().isoformat(), "metrics": tok
                    })
            save_json(TRACKING_PATH, pending_tracks)

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
        "model_location": str(MODEL_PATH) if MODEL_PATH.exists() else "None",
        "history_records": len(load_json(DATASET_PATH)),
        "active_tracks": len(load_json(TRACKING_PATH)),
        "paper_portfolio": load_portfolio()
    }

@app.get("/get_history")
def get_history():
    if DATASET_PATH.exists():
        return FileResponse(DATASET_PATH, media_type="application/json", filename="pump_history.json")
    return {"error": "File pump_history.json not found"}

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    dest_path = BASE_DIR / "pump_radar_model.json"
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "message": f"Модель успішно завантажено в {dest_path}"}

@app.post("/reset_portfolio")
def reset_portfolio():
    default_portfolio = {
        "balance": 10000.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_realized_pnl": 0.0
    }
    save_portfolio(default_portfolio)
    return {"status": "success", "message": "Портфель успішно скинуто до $10,000", "portfolio": default_portfolio}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)