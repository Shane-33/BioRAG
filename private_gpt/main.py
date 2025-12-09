"""FastAPI app creation, logger configuration and main API routes."""

from private_gpt.di import global_injector
from private_gpt.launcher import create_app

app = create_app(global_injector)

# -------------------------------
# Prevent UI from exiting immediately
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    import os

    # FastAPI + mounted Gradio UI
    print("\n Starting BioRAG server...")
    print(" UI: http://localhost:7860\n")

    uvicorn.run(
        "private_gpt.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )
