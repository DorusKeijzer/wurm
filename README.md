# wurm

This project's dependencies are managed by [Poetry](https://python-poetry.org/)

# Using Poetry to Run main.py

## Installing Poetry

1. Open your terminal or command prompt.

2. Install Poetry using one of these methods:

   - On macOS, Linux, or Windows (WSL):
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```

   - On Windows (PowerShell):
     ```powershell
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
     ```

3. Verify the installation:
   ```bash
   poetry --version
   ```

## Running main.py with Poetry
In the root directory:

1. install dependencies (this takes a while):
   ```bash
   poetry install
   ```
2. Run `main.py` using Poetry:
   ```bash
   poetry run python main.py
   ```


