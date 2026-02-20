# Plugin System

RRG AI supports local plugins so users can extend behavior without editing core backend code.

## Plugin Location

Default root:

- `plugins/`

Override with:

- `AI_PLUGINS_DIR=/path/to/plugins`

## Plugin Folder Structure

Each plugin must be in its own folder with a manifest and entrypoint script.

Example:

```text
plugins/
  text_tools/
    plugin.json
    run.py
```

## Manifest (`plugin.json`)

Required/commonly used fields:

```json
{
  "id": "text_tools",
  "name": "Text Tools",
  "version": "1.0.0",
  "description": "Tokenize text and produce lexical stats.",
  "entrypoint": "run.py",
  "timeout_sec": 30,
  "enabled": true,
  "allow_in_agent": true,
  "input_schema": {
    "type": "object",
    "description": "Provide text in input or input.text"
  }
}
```

## Runtime Contract

Plugin entrypoint receives JSON on stdin:

```json
{
  "input": "or object payload",
  "context": {"session_id": "..."},
  "plugin": {"plugin_id": "...", "name": "...", "version": "..."}
}
```

Plugin should write JSON to stdout:

```json
{
  "status": "ok",
  "summary": "short status line",
  "text": "detailed output",
  "provenance": [{"source": "plugin:example", "snippet": "..."}],
  "artifacts": []
}
```

If plugin outputs plain text, backend still captures it.

## API

- `GET /api/plugins` lists discovered plugins
- `POST /api/plugins/run` runs one plugin with payload

## Agent Routing

Agent supports plugin commands in natural prompts:

- `list plugins`
- `run plugin <plugin_id> with <input>`
- `plugin <plugin_id>: <input>`

## Safety Notes

- Plugins are local code and run on your machine.
- Treat plugins as trusted code only.
- Keep plugin directories under your control.
- Use conservative plugin timeouts (`timeout_sec`).
