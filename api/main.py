import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - handled via runtime exception
    OpenAI = None  # type: ignore

try:
    import resource  # type: ignore
except ImportError:  # pragma: no cover - resource is POSIX only
    resource = None  # type: ignore


class AgentConfigurationError(Exception):
    """Raised when the agent is misconfigured."""


class DangerousTaskError(Exception):
    """Raised when the task requests an unsafe action."""


class AgentExecutionError(Exception):
    """Raised when the agent fails to complete the task."""


class AgentInputError(Exception):
    """Raised when the provided task input is invalid."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        super().__init__(message)
        self.status_code = status_code


logger = logging.getLogger("tds-vercel-test")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_ALLOWED_ORIGIN = "https://example.com"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [DEFAULT_ALLOWED_ORIGIN]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

LATENCY_DATA_PATH = Path(
    os.getenv("LATENCY_DATA_PATH", Path(__file__).resolve().parent / "q-vercel-latency.json")
)


def _load_latency_records() -> List[Dict[str, Any]]:
    """Load latency records from disk; return an empty list if unavailable."""
    if not LATENCY_DATA_PATH.exists():
        logger.warning("Latency data file not found at %s", LATENCY_DATA_PATH)
        return []
    try:
        with LATENCY_DATA_PATH.open("r", encoding="utf-8") as f:
            records = json.load(f)
            if not isinstance(records, list):
                logger.error("Latency data file has unexpected structure.")
                return []
            return records
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode latency data: %s", exc)
        return []


def compute_metrics_for_region(records: List[Dict[str, Any]], threshold_ms: float) -> Dict[str, Optional[float]]:
    """Compute aggregate latency and uptime metrics for the provided records."""
    latencies = np.array([r["latency_ms"] for r in records if "latency_ms" in r], dtype=float)
    uptimes = np.array([r["uptime_pct"] for r in records if "uptime_pct" in r], dtype=float)
    return {
        "avg_latency": float(np.mean(latencies)) if latencies.size else None,
        "p95_latency": float(np.percentile(latencies, 95, method="nearest")) if latencies.size else None,
        "avg_uptime": float(np.mean(uptimes)) if uptimes.size else None,
        "breaches": int(np.sum(latencies > threshold_ms)) if latencies.size else 0,
    }


@app.post("/metrics")
async def metrics(request: Request) -> Dict[str, Dict[str, Optional[float]]]:
    try:
        payload = await request.json()
    except Exception as exc:  # pragma: no cover - FastAPI will surface this
        logger.error("Invalid JSON payload: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    regions = payload.get("regions")
    threshold_ms = payload.get("threshold_ms")

    if not isinstance(regions, list) or not all(isinstance(r, str) for r in regions):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="regions must be an array of strings")
    if not isinstance(threshold_ms, (int, float)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="threshold_ms must be a number")

    records = _load_latency_records()
    threshold_ms = float(threshold_ms)
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for region in {r.lower() for r in regions}:
        region_records = [r for r in records if str(r.get("region", "")).lower() == region]
        result[region] = compute_metrics_for_region(region_records, threshold_ms)
    return result


def _validate_api_key(header_value: Optional[str]) -> None:
    configured_key = os.getenv("AGENT_API_KEY")
    if not configured_key:
        raise AgentConfigurationError("Agent API key is not configured.")
    if header_value != configured_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


class AgentRunner:
    """Run a single supervised LLM agent execution within a restricted workspace."""

    DANGEROUS_PATTERNS = (
        " rm ",
        " rm -",
        "sudo ",
        "curl ",
        "wget ",
        "http://",
        "https://",
        "ssh ",
        "apt-get",
        "brew ",
        "pip install",
    )

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL")
        if not self.api_key or not self.model:
            raise AgentConfigurationError("OpenAI configuration missing.")
        if OpenAI is None:
            raise AgentConfigurationError("openai package is not available.")

        self.workspace_root = Path(os.getenv("AGENT_WORKSPACE_ROOT", "/tmp/agent-workspace"))
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.workspace_root, 0o700)

        self.log_dir = Path(os.getenv("AGENT_LOG_DIR", "/tmp/agent-logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.log_dir, 0o700)

        self.max_execution_seconds = int(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
        self.max_file_size = int(os.getenv("AGENT_MAX_FILE_SIZE", str(1024 * 1024)))
        self.max_memory_mb = int(os.getenv("AGENT_MAX_MEMORY_MB", "512"))
        self.client = OpenAI(api_key=self.api_key)

    def run(self, task: str) -> str:
        sanitized = task.strip()
        if not sanitized:
            raise AgentInputError("Task must not be empty.", status.HTTP_400_BAD_REQUEST)
        if len(sanitized) > 2000:
            raise AgentInputError("Task is too long.", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        lower_task = f" {sanitized.lower()} "
        if any(pattern in lower_task for pattern in self.DANGEROUS_PATTERNS):
            raise DangerousTaskError("Task rejected: contains potentially dangerous instructions.")

        run_id = uuid.uuid4().hex
        workspace = Path(tempfile.mkdtemp(prefix=f"run-{run_id}-", dir=self.workspace_root))
        log_file = self.log_dir / f"{run_id}.jsonl"
        self._initialize_log(log_file)

        logger.info("Agent run %s started in %s", run_id, workspace)
        start_time = time.monotonic()
        limits_applied = False

        try:
            if resource is not None:
                limits_applied = self._apply_resource_limits()
            self._append_log(log_file, {"event": "start", "run_id": run_id, "workspace": str(workspace)})

            prompt = (
                "You are a supervised coding agent executing a single automated run inside a sandbox. "
                "Perform the requested task and return your final answer succinctly. "
                "Do not ask for human input and do not attempt network access or shell commands that touch outside the workspace."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": sanitized},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            output = response.choices[0].message.content.strip() if response.choices else ""
            self._append_log(
                log_file,
                {
                    "event": "llm_response",
                    "run_id": run_id,
                    "tokens_used": response.usage.total_tokens if hasattr(response, "usage") else None,
                },
            )

            if not output:
                raise AgentExecutionError("Agent did not return any output.")

            duration = time.monotonic() - start_time
            if duration > self.max_execution_seconds:
                raise AgentExecutionError("Agent exceeded the allowed execution time.")

            self._append_log(log_file, {"event": "complete", "run_id": run_id, "duration_seconds": duration})
            return output
        except DangerousTaskError:
            self._append_log(log_file, {"event": "dangerous_task_rejected", "run_id": run_id})
            raise
        except AgentInputError:
            raise
        except Exception as exc:
            self._append_log(
                log_file,
                {"event": "error", "run_id": run_id, "message": str(exc), "type": exc.__class__.__name__},
            )
            raise AgentExecutionError("Agent execution failed.") from exc
        finally:
            if resource is not None and limits_applied:
                self._restore_resource_limits()
            try:
                shutil.rmtree(workspace)
            except Exception as exc:
                logger.warning("Failed to remove workspace %s: %s", workspace, exc)
            else:
                logger.debug("Workspace %s removed", workspace)

    def _initialize_log(self, log_path: Path) -> None:
        log_path.touch(exist_ok=True)
        os.chmod(log_path, 0o600)

    def _append_log(self, log_path: Path, entry: Dict[str, Any]) -> None:
        line = json.dumps(entry, separators=(",", ":"))
        if len(line.encode("utf-8")) > self.max_file_size:
            logger.warning("Log entry too large; truncating.")
            line = line[: self.max_file_size]
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            logger.error("Failed to write agent log: %s", exc)

    def _apply_resource_limits(self) -> bool:
        if resource is None:
            return False
        self._previous_limits: Dict[int, Any] = {}
        try:
            cpu_seconds = self.max_execution_seconds
            mem_bytes = self.max_memory_mb * 1024 * 1024
            try:
                self._previous_limits[resource.RLIMIT_CPU] = resource.getrlimit(resource.RLIMIT_CPU)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                logger.debug("Unable to set CPU limit.")
            try:
                self._previous_limits[resource.RLIMIT_AS] = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                logger.debug("Unable to set memory limit.")
            return True
        except Exception as exc:
            logger.debug("Failed to apply resource limits: %s", exc)
            return False

    def _restore_resource_limits(self) -> None:
        if resource is None or not hasattr(self, "_previous_limits"):
            return
        for limit, previous in self._previous_limits.items():
            try:
                resource.setrlimit(limit, previous)
            except Exception:
                logger.debug("Failed to restore resource limit %s", limit)


async def authenticate(api_key: Optional[str] = Header(None, alias="x-api-key")) -> None:
    try:
        _validate_api_key(api_key)
    except AgentConfigurationError as exc:
        logger.error("Authentication configuration error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.get("/task")
async def run_task(
    q: str = Query(..., min_length=1, max_length=2000),
    _: None = Depends(authenticate),
) -> Dict[str, str]:
    runner = AgentRunner()

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(runner.run, q),
            timeout=runner.max_execution_seconds + 5,
        )
    except DangerousTaskError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AgentInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except AgentConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except AgentExecutionError as exc:
        logger.error("Agent execution error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        logger.error("Agent run timed out.")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Agent timed out.") from exc

    response = {
        "task": q,
        "agent": "copilot-cli",
        "output": result,
        "email": "21f1006362@ds.study.iitm.ac.in",
    }
    return response
