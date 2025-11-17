from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Columns
TEXT_COLS = ["why", "what"]      # from evaluate_LG_on_D2
CODE_COLS = ["industry_code"]    # optional numeric feature
TARGET_COL = "mjj2cclean"        # same label column used in evaluate_LG_on_D2

# Model settings
CV_FOLDS = 5
RANDOM_STATE = 42



import logging
from pathlib import Path
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    log_file = output_dir / "run.log"

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Logging initialized.")



from pathlib import Path
import logging
import pandas as pd
from coder.config import DATA_DIR

def resolve_train_file():
    candidates = [
        DATA_DIR / "NLFS_2024Q1_INDIVIDUAL.xlsx",
        DATA_DIR / "NLFS_2024Q1_INDIVIDUAL.xls",
    ]
    for p in candidates:
        if p.exists():
            logging.info(f"Using TRAIN file: {p}")
            return p
    raise FileNotFoundError("Training file not found in /data folder.")

def resolve_test_file():
    candidates = [
        DATA_DIR / "NLFS_2024_Q2.csv",
        DATA_DIR / "NLFS_2024_Q2.txt",
    ]
    for p in candidates:
        if p.exists():
            logging.info(f"Using TEST file: {p}")
            return p
    raise FileNotFoundError("Test file not found in /data folder.")

def load_data():
    train_path = resolve_train_file()
    test_path = resolve_test_file()

    df_train = pd.read_excel(train_path, engine="openpyxl")
    df_test = pd.read_csv(test_path)

    return df_train, df_test



import logging
import pandas as pd
from coder.config import TEXT_COLS, CODE_COLS, TARGET_COL

def normalize_code_str(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace(".0", "")
    digits = "".join([c for c in s if c.isdigit()])
    return digits.zfill(4) if digits else s

def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # Clean training
    df_train = df_train.dropna(subset=[TARGET_COL]).copy()
    df_train[TEXT_COLS] = df_train[TEXT_COLS].fillna("")

    df_train["combined_text"] = df_train[TEXT_COLS].apply(
        lambda row: " ".join(row.astype(str)), axis=1
    )

    # Clean test
    df_test = df_test.copy()
    df_test[TEXT_COLS] = df_test[TEXT_COLS].fillna("")
    df_test["combined_text"] = df_test[TEXT_COLS].apply(
        lambda row: " ".join(row.astype(str)), axis=1
    )

    X_train = df_train[["combined_text"] + CODE_COLS]
    y_train = df_train[TARGET_COL].astype(str).map(normalize_code_str)

    X_test = df_test[["combined_text"] + CODE_COLS]
    y_test = df_test[TARGET_COL].astype(str).map(normalize_code_str) if TARGET_COL in df_test else None

    logging.info(f"Training rows: {len(df_train)}, Test rows: {len(df_test)}")

    return X_train, y_train, X_test, y_test



import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():
    logging.info("Building TF-IDF + Logistic Regression model")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])
    return pipe



import pandas as pd
from sklearn.metrics import classification_report
from coder.preprocessing import normalize_code_str

def evaluate_model(model, X, y_true):
    pred_raw = model.predict(X)
    y_pred = pd.Series(pred_raw).map(normalize_code_str)

    rep = classification_report(
        y_true, y_pred, digits=4, zero_division=0, output_dict=True
    )
    return pd.DataFrame(rep).transpose()



import logging
from coder.utils import setup_logging, timestamp
from coder.data_utils import load_data
from coder.preprocessing import prepare_data
from coder.model import build_model
from coder.evaluate import evaluate_model
from coder.config import OUTPUT_DIR

def main():
    setup_logging(OUTPUT_DIR)
    logging.info("Starting simple ISCO classifier pipeline...")

    # Load
    df_train, df_test = load_data()

    # Prepare
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

    # Train
    model = build_model()
    model.fit(X_train, y_train)

    # --- D1 report ---
    rep_d1 = evaluate_model(model, X_train, y_train)
    rep_d1.to_csv(OUTPUT_DIR / f"classification_report_D1.csv", index=True)
    logging.info("Saved D1 report.")

    # --- D2 predictions & report ---
    preds = model.predict(X_test)
    df_out = df_test.copy()
    df_out["pred_mjj2cclean"] = preds
    df_out.to_csv(OUTPUT_DIR / f"predictions_D2.csv", index=False)
    logging.info("Saved D2 predictions.")

    if y_test is not None:
        rep_d2 = evaluate_model(model, X_test, y_test)
        rep_d2.to_csv(OUTPUT_DIR / f"classification_report_D2.csv", index=True)
        logging.info("Saved D2 report.")

    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main()


