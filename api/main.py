import json
import os
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for POST from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load dataset once at startup; file placed at project root
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "q-vercel-latency.json")
if not os.path.exists(DATA_PATH):
    # Support local dev run: also try current working directory
    alt = os.path.join(os.getcwd(), "q-vercel-latency.json")
    DATA_PATH = alt if os.path.exists(alt) else DATA_PATH

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        RECORDS: List[Dict[str, Any]] = json.load(f)
except Exception as e:
    # Fail fast with clear error; Vercel logs will show this
    RECORDS = []
    print(f"Failed to load dataset: {e}")

def compute_metrics_for_region(records: List[Dict[str, Any]], threshold_ms: float) -> Dict[str, float]:
    latencies = np.array([r["latency_ms"] for r in records], dtype=float)
    uptimes = np.array([r["uptime_pct"] for r in records], dtype=float)

    avg_latency = float(np.mean(latencies)) if latencies.size else None
    p95_latency = float(np.percentile(latencies, 95, method="nearest")) if latencies.size else None
    avg_uptime = float(np.mean(uptimes)) if uptimes.size else None
    breaches = int(np.sum(latencies > threshold_ms)) if latencies.size else 0

    return {
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "avg_uptime": avg_uptime,
        "breaches": breaches,
    }

@app.post("/metrics")
async def metrics(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Validate input
    regions = payload.get("regions")
    threshold_ms = payload.get("threshold_ms")
    if not isinstance(regions, list) or not all(isinstance(r, str) for r in regions):
        raise HTTPException(status_code=400, detail="regions must be an array of strings")
    if not (isinstance(threshold_ms, int) or isinstance(threshold_ms, float)):
        raise HTTPException(status_code=400, detail="threshold_ms must be a number")

    # Normalize regions to lowercase for matching
    regions = [r.lower() for r in regions]

    # Group records per region (case-insensitive)
    result = {}
    for region in set(regions):
        region_records = [r for r in RECORDS if str(r.get("region", "")).lower() == region]
        result[region] = compute_metrics_for_region(region_records, float(threshold_ms))

    return result
