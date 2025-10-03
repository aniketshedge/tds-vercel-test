from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os
import numpy as np
from typing import List, Dict, Any

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.getcwd(), "q-vercel-latency.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    RECORDS: List[Dict[str, Any]] = json.load(f)

def compute_metrics_for_region(records: List[Dict[str, Any]], threshold_ms: float) -> Dict[str, float]:
    latencies = np.array([r["latency_ms"] for r in records], dtype=float)
    uptimes = np.array([r["uptime_pct"] for r in records], dtype=float)
    return {
        "avg_latency": float(np.mean(latencies)) if latencies.size else None,
        "p95_latency": float(np.percentile(latencies, 95, method="nearest")) if latencies.size else None,
        "avg_uptime": float(np.mean(uptimes)) if uptimes.size else None,
        "breaches": int(np.sum(latencies > threshold_ms)) if latencies.size else 0,
    }

@app.post("/")
async def metrics(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    regions = payload.get("regions")
    threshold_ms = payload.get("threshold_ms")
    if not isinstance(regions, list) or not all(isinstance(r, str) for r in regions):
        raise HTTPException(status_code=400, detail="regions must be an array of strings")
    if not isinstance(threshold_ms, (int, float)):
        raise HTTPException(status_code=400, detail="threshold_ms must be a number")
    regions = [r.lower() for r in regions]
    result = {}
    for region in set(regions):
        region_records = [r for r in RECORDS if str(r.get("region", "")).lower() == region]
        result[region] = compute_metrics_for_region(region_records, float(threshold_ms))
    return result
