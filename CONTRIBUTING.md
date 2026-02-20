# Contributing

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run backend locally:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

## Tests

```bash
PYTHONPATH=. pytest -q
```

## Coding Rules

- Keep changes local-first (no mandatory cloud dependencies).
- Do not break existing API endpoints.
- Prefer additive changes for response payloads; avoid removing established fields.
- Add tests for new behavior.
- Keep security defaults conservative (localhost, token auth, root confinement).

## Documentation Rules

If you change behavior in any of these areas, update docs in the same PR:

- API behavior -> `docs/API_REFERENCE.md`
- security/auth/cors/files/web policy -> `docs/SECURITY.md`
- startup/config -> `README.md` and `docs/OPERATIONS.md`
- eval/bench/report format -> `docs/BENCHMARKS.md`

## Pull Request Checklist

- [ ] Tests pass (`PYTHONPATH=. pytest -q`)
- [ ] No secrets or `.ai_data/*` files included
- [ ] Docs updated for any behavior changes
- [ ] Backward compatibility preserved for existing endpoints
