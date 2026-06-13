# Contributing

Contributions are welcome! Please follow these guidelines to ensure a smooth collaboration process.

## Getting Started

1. Fork the repository and create your branch from `main`.
2. Follow the instructions in `SETUP.md` to configure your environment and download the dataset.
3. Install the development dependencies:
   ```bash
   make install
   ```

## Development Workflow

1. **Write Code:** Ensure your code is clean, modular, and well-documented.
2. **Type Hints:** All new functions and methods must include Python type hints.
3. **Linting and Formatting:** We use `ruff` for code quality. Run `make lint` before committing to ensure there are no errors and the code is formatted correctly.
4. **Notebooks:** The `.pre-commit-config.yaml` automatically runs `nbstripout` to clear notebook outputs. Do not commit notebook outputs, as this bloats the git history.

## Pull Requests

- Keep PRs focused on a single feature or bug fix.
- Ensure all tests pass (`make test`).
- Update the `README.md` or other documentation if your change affects user-facing features or architecture.
