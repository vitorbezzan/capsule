# Capsule

[![python](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge)](http://python.org)
[![python](https://img.shields.io/badge/python-3.13-blue?style=for-the-badge)](http://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)


Capsule is a Python library for machine learning tasks, providing a unified interface for classification and regression models. The source code is organized in the `src/capsule/` directory, with modules for base functionality, classification, and regression.

## Features
- Unified API for classification and regression
- Extensible base classes
- Easy integration with other Python ML libraries

## Installation

### Install from GitHub
```bash
pip install git+https://github.com/vitorbezzan/capsule.git
```

## Development Setup

To set up Capsule for development, follow these detailed steps:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/capsule.git
cd capsule
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n capsule python=3.8+
conda activate capsule
```

### 3. Install in Development Mode
This project uses `pyproject.toml` for dependency management. Install the package in editable mode with all dependencies:

```bash
# Install the package in development mode
pip install -e .

# Install with development dependencies (if specified in pyproject.toml)
pip install -e ".[dev]"
```

### 4. Verify Installation
```bash
# Run tests to verify everything is working
pytest

# Or run tests with coverage
pytest --cov=capsule
```

### 5. Development Workflow
- The source code is located in `src/capsule/`
- Tests are in the `tests/` directory
- Documentation files are in `docs/`
- Use `pytest` to run tests during development
- The project configuration is managed through `pyproject.toml`

## License
See [LICENSE](LICENSE) for details.
