# ApexView

Headless dental radiograph processing engine.

## Architecture

The core principle of ApexView is a strict separation between computation and
presentation:

- **The engine is a headless library that owns ALL computation.** Every
  measurement, classification, transformation, and derived value is produced
  inside the engine.
- **Any future UI is a thin client.** The UI only displays what the engine
  returns. It never recomputes, re-derives, or second-guesses engine values.

This keeps the engine independently testable, embeddable in any host, and the
single source of truth for all radiograph-derived data.

## Development setup

```
pip install -e ".[dev]"
```

## Running tests

```
pytest
```
