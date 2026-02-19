# RRG AI (GitHub Pages)

Static HTML/JavaScript AI chat experience designed to run directly on GitHub Pages.

## Features

- Chat UI with persistent browser memory (localStorage)
- Natural language tool inference:
  - math: "calculate (12+8)/5"
  - repo-style search across built-in corpus: "search entropy_rag in repo"
  - stats: "how many docs"
- Planning mode for optimization/system prompts
- No backend required for Pages

## Deploy on GitHub Pages

1. Push this repository to GitHub.
2. In GitHub repo settings:
   - Open `Pages`
   - Source: `Deploy from a branch`
   - Branch: `main` and folder `/ (root)`
3. Save and wait for the Pages URL.

## Notes

GitHub Pages is static hosting. Python server components do not execute on Pages.
