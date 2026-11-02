import os
import threading
from datetime import datetime, timezone
import pandas as pd

EXCEL_FILE_PATH = os.path.join("data", "llm_usage_and_audit_logs.xlsx")
_excel_lock = threading.Lock()

REQUEST_AUDIT_COLUMNS = [
    "timestamp",
    "request_id",
    "user_query",
    "dbml_input",
    "summary_enabled",
    "model_used",
    "status",
    "http_status_code",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "dbml_output",
    "error_message",
]

LIVE_USAGE_COLUMNS = [
    "date",
    "model_name",
    "total_requests",
    "total_tokens",
    "last_used_at",
]


def _ensure_excel_file():
    """Ensure Excel file exists with proper sheets and columns."""
    os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)
    if not os.path.exists(EXCEL_FILE_PATH):
        with pd.ExcelWriter(EXCEL_FILE_PATH, engine="openpyxl") as writer:
            pd.DataFrame(columns=REQUEST_AUDIT_COLUMNS).to_excel(
                writer, sheet_name="Request_Audit_Logs", index=False
            )
            pd.DataFrame(columns=LIVE_USAGE_COLUMNS).to_excel(
                writer, sheet_name="Live_Usage_Summary", index=False
            )


def log_request_to_excel(
    request_id: str,
    user_query: str,
    dbml_input: str,
    summary_enabled: bool,
    model_used: str,
    status: str,
    http_status_code: int,
    duration_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    dbml_output: str | None = None,
    error_message: str | None = None,
):
    """Append a request audit row and update live usage summary in Excel."""
    with _excel_lock:
        try:
            _ensure_excel_file()
            now_utc = datetime.now(timezone.utc)
            now_iso = now_utc.isoformat()
            today_str = now_utc.strftime("%Y-%m-%d")

            # 1. Read existing sheets
            audit_df = pd.read_excel(
                EXCEL_FILE_PATH, sheet_name="Request_Audit_Logs", engine="openpyxl"
            )
            usage_df = pd.read_excel(
                EXCEL_FILE_PATH, sheet_name="Live_Usage_Summary", engine="openpyxl"
            )

            # 2. Add audit row
            new_audit_row = pd.DataFrame(
                [
                    {
                        "timestamp": now_iso,
                        "request_id": str(request_id),
                        "user_query": user_query,
                        "dbml_input": dbml_input or "",
                        "summary_enabled": summary_enabled,
                        "model_used": model_used,
                        "status": status,
                        "http_status_code": http_status_code,
                        "duration_ms": duration_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "dbml_output": dbml_output or "",
                        "error_message": error_message or "",
                    }
                ]
            )
            audit_df = pd.concat([audit_df, new_audit_row], ignore_index=True)

            # 3. Update or insert live daily usage
            mask = (usage_df["date"] == today_str) & (usage_df["model_name"] == model_used)
            if mask.any():
                idx = usage_df[mask].index[0]
                usage_df.at[idx, "total_requests"] += 1
                usage_df.at[idx, "total_tokens"] += total_tokens
                usage_df.at[idx, "last_used_at"] = now_iso
            else:
                new_usage_row = pd.DataFrame(
                    [
                        {
                            "date": today_str,
                            "model_name": model_used,
                            "total_requests": 1,
                            "total_tokens": total_tokens,
                            "last_used_at": now_iso,
                        }
                    ]
                )
                usage_df = pd.concat([usage_df, new_usage_row], ignore_index=True)

            # 4. Save both sheets back to Excel
            with pd.ExcelWriter(EXCEL_FILE_PATH, engine="openpyxl", mode="w") as writer:
                audit_df.to_excel(writer, sheet_name="Request_Audit_Logs", index=False)
                usage_df.to_excel(writer, sheet_name="Live_Usage_Summary", index=False)

        except Exception as e:
            # We don't want logging errors to crash the user's API response
            import logging
            logging.getLogger(__name__).error(f"Failed to write to Excel audit log: {e}")

