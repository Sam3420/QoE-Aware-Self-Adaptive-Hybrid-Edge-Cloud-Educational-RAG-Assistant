# QoE Educational Assistant

Phase 1 implements the project foundation and core data layer only.

## Local Setup

```bash
python -m pip install -e ".[test]"
pytest
```

Configuration is loaded from environment variables and optional `.env` values. See `.env.example` for local edge deployment defaults.
