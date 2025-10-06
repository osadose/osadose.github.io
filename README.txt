# ---- File paths
DATA_DIR = "data"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "NLFS_2024Q1_INDIVIDUAL 2.xlsx")
TRAIN_FILE_FALLBACK = os.path.join(DATA_DIR, "NLFS_2024Q1_INDIVIDUAL.xlsx")
TEST_FILE = os.path.join(DATA_DIR, "NLFS_2024_Q2.csv")


# 1) D1 hold-out report
out_path = os.path.join(OUTPUT_DIR, "classification_report_D1_filtered_holdout.csv")
pd.DataFrame(report_holdout).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path, index=False)
print(f"[Info] classification_report saved to: {out_path}")

# 2) D2 predictions
pred_path = os.path.join(OUTPUT_DIR, "predictions_D2_from_LG.csv")
out.to_csv(pred_path, index=False)
print(f"\n[Info] Saved predictions for ALL D2 rows to: {pred_path}")

# 3) D2 classification report
rep_d2_path = os.path.join(OUTPUT_DIR, "classification_report_D2_full.csv")
pd.DataFrame(report_d2).transpose().reset_index().rename(columns={"index":"label"}).to_csv(rep_d2_path, index=False)
print(f"[Info] classification_report saved to: {rep_d2_path}")

