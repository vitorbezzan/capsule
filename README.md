# Capsule

[![python](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge)](http://python.org)
[![python](https://img.shields.io/badge/python-3.13-blue?style=for-the-badge)](http://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Capsule** is a production-ready Python library that wraps trained models with
built-in drift detection, performance monitoring, and visualization for both
classification and regression tasks.

Unlike traditional ML libraries that focus on training, Capsule specializes in production deployment concerns—wrapping your trained models with enterprise-grade monitoring and security features while maintaining a clean, unified API.

## ✨ Key Features

- **📊 Built-in Drift Detection**: Automatic univariate drift monitoring using NannyML
- **📈 Performance Monitoring**: CBPE (classification) and DLE (regression)
- **🎨 Visualizations**: ROC/PR curves for classification; scatter/residuals for regression
- **🎯 Unified API**: Consistent interface for both classification and regression models
- **🔧 Framework Agnostic**: Works with any scikit-learn compatible model
- **🛡️ Immutable Wrappers**: Prevents accidental model modification
- **📦 Type Safe**: Type hints plus runtime validation with Pydantic

## 🚀 Quick Start

```python
from capsule.classification import ClassificationCapsule
from capsule.regression import RegressionCapsule
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train your model as usual
model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)

# Wrap it in a Capsule
capsule = ClassificationCapsule(model, X_test, y_test)

# Make predictions
predictions = capsule.predict(X_new)

# Visualize performance (uses stored test data by default)
capsule.plots.roc_curve()

# Monitor for drift on production-like data
drift_results = capsule.univariate_drift(X_prod)

# Estimate performance on unlabeled production data
perf = capsule.metrics(X_prod, metric="f1")
```

## Installation

### Install from GitHub
```bash
pip install git+https://github.com/vitorbezzan/capsule.git
```

## Development Setup

To set up Capsule for development, follow these detailed steps:

### 1. Clone the Repository
```bash
git clone https://github.com/vitorbezzan/capsule.git
cd capsule
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n capsule python=3.11+
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
