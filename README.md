# wurm

This project's dependencies are managed by [Poetry](https://python-poetry.org/)

# Using Poetry to Run main.py

## Installing Poetry

1. Open your terminal or command prompt.

2. Install Poetry using one of these methods:

   - On macOS, Linux, or Windows (WSL):
     ```sh
     curl -sSL https://install.python-poetry.org | python3 -
     ```

   - On Windows (PowerShell):
     ```powershell
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
     ```

3. Verify the installation:
   ```sh
   poetry --version
   ```

## Running main.py with Poetry
1. install dependencies:
   ```sh
   poetry install
   ```
2. Run `main.py` using Poetry:
   ```sh
   poetry run python main.py
   ```


