from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import LocalAgent
from .llm import OllamaClient
from .plugins import PluginManager
from .storage import SQLiteStore
from .tools.filesystem import FileBrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="RRG AI terminal chat")
    parser.add_argument("--db", default=".ai_data/ai.sqlite3")
    parser.add_argument("--downloads", default=".ai_data/downloads")
    parser.add_argument("--files-root", default=os.getenv("AI_FILES_ROOT", str(Path.home())))
    parser.add_argument("--plugins-root", default=os.getenv("AI_PLUGINS_DIR", "plugins"))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument(
        "--recursive-adic-ranking",
        default=os.getenv("AI_RECURSIVE_ADIC_RANKING", "1"),
        help="Enable Recursive-Adic chunk ranking (1/0).",
    )
    parser.add_argument(
        "--radf-beta",
        default=os.getenv("AI_RADF_BETA", "0.35"),
        help="Depth-Laplace beta for Recursive-Adic ranking.",
    )
    parser.add_argument(
        "--radf-alpha",
        default=os.getenv("AI_RADF_ALPHA", "1.5"),
        help="Recursive depth alpha for RDT recurrence.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    use_recursive_adic = str(args.recursive_adic_ranking).lower() not in {"0", "false", "no"}
    store = SQLiteStore(
        root / args.db,
        use_recursive_adic=use_recursive_adic,
        radf_beta=float(args.radf_beta),
        radf_alpha=float(args.radf_alpha),
    )
    files = FileBrowser(Path(args.files_root))
    plugins_root = Path(args.plugins_root).expanduser()
    if not plugins_root.is_absolute():
        plugins_root = (root / plugins_root).resolve()
    plugins = PluginManager(plugins_root)
    llm = OllamaClient(model=args.model, base_url=args.ollama_url)
    agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=root / args.downloads, plugins=plugins)

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
