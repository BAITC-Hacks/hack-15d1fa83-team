# Project workspace

The user requires all project work to stay in this repository directory (`Source`).

- Use this directory as the working directory for commands and edits.
- Keep the virtual environment at `.venv/` inside this directory.
- Keep local input documents in `docs/source-materials/` and generated outputs in `output/`.
- Do not create project files or environments in the parent `Hackalem AI` directory.
- The unified release targets `main`; prepare and test merges on an integration branch first. Preserve the original feature branches and unrelated changes.
- Backend: Django / Python. Frontend: native HTML, CSS and JavaScript, without frontend frameworks.
- ML integration contract: `docs/ML_INTEGRATION.md`. Do not substitute observations or reanalysis for archived weather forecasts.

Run checks from this directory with `.venv/Scripts/python.exe manage.py check` and `.venv/Scripts/python.exe manage.py test` on Windows.
