@echo off
echo --- SEARCH22.5: Initialization ---


:: 1. Create the virtual environment using uv
echo [1/3] Creating virtual environment...
uv venv

:: 2. Sync dependencies and compile C++ extension
:: uv pip install -e . looks at pyproject.toml and builds the ext_modules
echo [2/3] Installing dependencies and compiling C++ Core...
uv pip install -e .

:: 3. Verification
echo [3/3] Verifying installation...
uv run python -c "import engine.math225_core; print('C++ Core Compiled Successfully!')"
echo ---------------------------------------------
echo Setup complete.
pause

