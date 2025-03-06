# rag_system/main.py
import argparse
import multiprocessing
import os
import platform
import webbrowser

import uvicorn

from core.config import settings


def start_api(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    uvicorn.run("api.main:app", host=host, port=port, reload=True)


def start_ui():
    """Start the Streamlit UI."""
    os.system("streamlit run ui/app.py")


def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="PDF RAG System")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--ui-only", action="store_true", help="Run only the UI server")
    parser.add_argument("--api-host", default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Run selected components
    if args.api_only:
        print("Starting API server...")
        start_api(host=args.api_host, port=args.api_port)
    elif args.ui_only:
        print("Starting UI server...")
        start_ui()
    else:
        # Run both components in separate processes
        print("Starting API and UI servers...")

        # Start API server
        api_process = multiprocessing.Process(
            target=start_api,
            args=(args.api_host, args.api_port)
        )
        api_process.start()

        # Give API server time to start
        import time
        time.sleep(2)

        # Open browser if requested
        if not args.no_browser:
            webbrowser.open("http://localhost:8501")

        # Start UI in main process
        start_ui()

        # Clean up API process when UI is closed
        api_process.terminate()
        api_process.join()


if __name__ == "__main__":
    # Handle Windows multiprocessing
    if platform.system() == "Windows":
        multiprocessing.freeze_support()

    main()