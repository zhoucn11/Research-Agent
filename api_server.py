"""Backward-compatible API server entrypoint."""

from research_agent.api.server import app


if __name__ == "__main__":
    import uvicorn

    print("[SYSTEM] Starting RESTful Agent Server on :8080")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=False)
