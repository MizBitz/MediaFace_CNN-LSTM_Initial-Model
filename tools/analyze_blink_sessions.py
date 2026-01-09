import argparse
import csv
import json
import os
from glob import glob
from typing import Any


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate mc_inference session summaries into a CSV report.")
    ap.add_argument("--log-dir", default="session_logs", help="Directory containing *_summary.json files")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <log-dir>/aggregate_report.csv)")
    args = ap.parse_args()

    log_dir = args.log_dir
    out_csv = args.out or os.path.join(log_dir, "aggregate_report.csv")

    summary_files = sorted(glob(os.path.join(log_dir, "mc_session_*_summary.json")))
    if not summary_files:
        raise SystemExit(f"No summary files found in: {log_dir}")

    rows: list[dict[str, Any]] = []

    for path in summary_files:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        session_id = summary.get("session_id")
        started_at = summary.get("started_at")
        ended_at = summary.get("ended_at")
        cfg = summary.get("config", {})
        eval_info = summary.get("eval", {})
        attempts = (eval_info or {}).get("attempts", [])

        cers = [safe_float(a.get("cer")) for a in attempts]
        wers = [safe_float(a.get("wer")) for a in attempts]
        cers_f = [c for c in cers if c is not None]
        wers_f = [w for w in wers if w is not None]

        mean_cer = (sum(cers_f) / len(cers_f)) if cers_f else None
        mean_wer = (sum(wers_f) / len(wers_f)) if wers_f else None

        rows.append(
            {
                "session_id": session_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "eval_enabled": bool(eval_info.get("enabled")),
                "attempts": len(attempts),
                "mean_cer": mean_cer,
                "mean_wer": mean_wer,
                "CNN_MODEL_PATH": cfg.get("CNN_MODEL_PATH"),
                "LSTM_MODEL_PATH": cfg.get("LSTM_MODEL_PATH"),
                "CHAR_PAUSE_THRESHOLD": cfg.get("CHAR_PAUSE_THRESHOLD"),
                "WORD_PAUSE_THRESHOLD": cfg.get("WORD_PAUSE_THRESHOLD"),
                "WORD_PAUSE_GRACE": cfg.get("WORD_PAUSE_GRACE"),
                "SENTENCE_SPEAK_BLINK_SEC": cfg.get("SENTENCE_SPEAK_BLINK_SEC"),
                "MIN_OPEN_STABILITY": cfg.get("MIN_OPEN_STABILITY"),
                "OPEN_PROB_THRESHOLD": cfg.get("OPEN_PROB_THRESHOLD"),
                "summary_path": path,
            }
        )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
