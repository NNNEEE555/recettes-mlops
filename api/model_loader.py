from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

global_payload = joblib.load(MODELS_DIR / "xgb_global.pkl")
segment_payload = joblib.load(MODELS_DIR / "xgb_segment.pkl")

global_model = global_payload["model"]
segment_model = segment_payload["pipeline"]