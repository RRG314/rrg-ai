from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import LocalAgent
from .llm import OllamaClient
from .storage import SQLiteStore
from .tools.filesystem import FileBrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="RRG AI terminal chat")
    parser.add_argument("--db", default=".ai_data/ai.sqlite3")
    parser.add_argument("--downloads", default=".ai_data/downloads")
    parser.add_argument("--files-root", default=os.getenv("AI_FILES_ROOT", str(Path.home())))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "llama3.1:8b"))
    parser.add_argument("--ollama-url", default=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"))
    args = parser.parse_args()

    root = Path.cwd()
    store = SQLiteStore(root / args.db)
    files = FileBrowser(Path(args.files_root))
    llm = OllamaClient(model=args.model, base_url=args.ollama_url)
    agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=root / args.downloads)

    session_id: str | None = None
    print("RRG AI terminal mode. Type 'exit' to quit.")

    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        result = agent.chat(session_id, user)
        session_id = str(result["session_id"])

        print(f"ai[{result['mode']}]> {result['answer']}")
        events = result.get("tool_events") or []
        if events:
            print("tools>")
            for event in events:
                print(f"  - [{event['status']}] {event['tool']}: {event['detail']}")


if __name__ == "__main__":
    main()
