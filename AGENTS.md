# Repository Guidelines

## Project

Renal Evidence Studio is an educational CKD screening demo built around the UCI Chronic Kidney Disease dataset. Keep medical language limited to risk screening and model behavior. Do not present outputs as clinical diagnosis.

## Commands

- Fetch/cache data: `python scripts/fetch_data.py`
- Train artifacts: `python -m src.train`
- Run API and UI: `uvicorn app.main:app --reload --port 8000`
- Run tests: `pytest -q`

## Constraints

- Do not copy names, README text, frontend flow, endpoint contracts, or UI styling from the Smart-Kidney-Sense reference repository.
- Keep the public API schema in English with semantic categorical values.
- Use one serialized scikit-learn `Pipeline` artifact for preprocessing plus model inference.
- Cite the UCI CKD dataset in user-facing documentation and model metadata.
- Add or update tests when behavior changes.
