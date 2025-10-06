from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import os

app = FastAPI(title="XBTUSD daily exporter")

CSV_PATH = os.getenv("CSV_PATH", "xbtusd_1h_8y.csv")

# ---------- your original load/resample logic ----------
def load_and_convert_to_daily(path=CSV_PATH):
    try:
        df = pd.read_csv(path)
        df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601")
        df = df.set_index("open_time").sort_index()

        daily = (
            df.resample("D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        return daily
    except FileNotFoundError:
        return None
    except Exception:
        return None

# ---------- download endpoint ----------
@app.get("/download/daily")
def download_daily():
    daily = load_and_convert_to_daily()
    if daily is None:
        return Response("Daily data not available", status_code=500)

    buf = io.StringIO()
    daily.to_csv(buf, index_label="date")
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="XBTUSD_daily.csv"'},
    )

# ---------- health check ----------
@app.get("/")
def root():
    return {"message": "XBTUSD daily exporter is up"}

# ---------- local dev only ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
