    # ---- Save D1 hold-out report to output folder ----
    if 'report_holdout' in locals():
        out_path = os.path.join(OUTPUT_DIR, "classification_report_D1_filtered_holdout.csv")
        pd.DataFrame(report_holdout).transpose().reset_index().rename(columns={"index": "label"}).to_csv(out_path, index=False)
        print(f"[Info] classification_report saved to: {out_path}")
    else:
        print("[Warn] No report_holdout generated — skipping save.")
