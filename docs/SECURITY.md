# Security Model

This project is local-first. The safest deployment pattern is to run backend and UI on localhost only.

## Security Defaults

From `backend/app.py` defaults:

- Bind host: `127.0.0.1` (via startup command)
- Auth required: `AI_REQUIRE_TOKEN=1`
- CORS allow origin regex: localhost-only by default
- Pairing required for non-local origins (`AI_BOOTSTRAP_PAIRING_REQUIRED=1`)
- File root default: repo root (`AI_FILES_ROOT`)

From `backend/tools/web.py`:

- Only `http`/`https` schemes allowed
- Blocks private/loopback/internal host patterns
- Blocks non-public resolved IPs to reduce SSRF risk

## Trust Boundaries

- Browser UI is untrusted until authenticated through token bootstrap.
- Backend is trusted local execution surface and can access local files/tools based on flags.
- Tool outputs are treated as untrusted content and should be grounded through provenance/evidence for factual responses.

## Auth Flow

1. Local UI calls `/api/bootstrap`.
2. If origin is local, token can be returned for local bootstrapping.
3. If origin is non-local and pairing is required, valid pairing code is needed.
4. Protected `/api/*` routes require header `x-ai-token`.

Exempt endpoints:
- `/api/health`
- `/api/bootstrap`
- `/api/pairing-code`

## File Access Safety

- File operations are rooted to `AI_FILES_ROOT`.
- Paths outside root are rejected.
- Use a dedicated minimal directory for `AI_FILES_ROOT` in shared machines.

## Recommended Hardening Profile

For strict local-only usage:

```bash
export AI_REQUIRE_TOKEN=1
export AI_ALLOWED_ORIGIN_REGEX='^https?://(127\\.0\\.0\\.1|localhost)(:\\d+)?$'
export AI_BOOTSTRAP_PAIRING_REQUIRED=1
export AI_FILES_ROOT="$(pwd)"
```

Optional additional controls:
- Disable web tools per request (`allow_web=false`) for offline or high-trust workflows.
- Disable code tools per request (`allow_code=false`) when not needed.
- Keep `allow_downloads=false` unless explicitly required.

## Operational Security Notes

- `.ai_data/` contains sensitive local data (token, memory DB, reports).
- `.ai_data/` is gitignored by default; keep it out of commits.
- Rotate API token by deleting `.ai_data/api_token.txt` and restarting backend.
- Regenerate pairing code by deleting `.ai_data/pairing_code.txt` and restarting backend.

## Known Limits

- This is not a sandboxed VM execution environment.
- `code.run` and `code.test` execute local processes under allowlist policy; treat as high-trust operations.
- If you expose backend beyond localhost, you must add network-layer controls (firewall/reverse-proxy/auth).
