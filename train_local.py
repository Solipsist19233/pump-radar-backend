import json
from pathlib import Path
import pandas as pd
import xgboost as xgb

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "pump_radar_model.json"
DATASET_PATH = PROJECT_DIR / "pump_history.json"


def train():
    if not DATASET_PATH.exists():
        print("[ERROR] Файл pump_history.json не знайдено!")
        print("Створіть тестові дані або додайте їх через API /add_history.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) < 10:
        print(
            f"[WARNING] У базі всього {len(data)} записів. Для нормального навчання треба хоча б 20-50 записів."
        )

    df = pd.DataFrame(data)

    feature_cols = ["mcap", "volume_5m", "buys_5m", "sells_5m", "top10_pct"]
    X = df[feature_cols]
    y = df["is_pump"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )

    print(f"Початок навчання на {len(df)} зразках...")
    model.fit(X, y)

    model.save_model(MODEL_PATH)
    print(f"✅ Навчання завершено! Модель збережена в: {MODEL_PATH}")


if __name__ == "__main__":
    train()