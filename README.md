# doombuoy
Use this command to install 

"pip install --extra-index https://test.pypi.org/simple/ doombuoy"

This library streamlines model comparison for data science projects.  
It currently supports binary classification, multiclass classification, regression, and time series tasks.  
With doombuoy, you can quickly generate performance summaries for a wide range of popular models, helping you select or set a baseline for your modeling efforts.

## Installation

```bash
$ pip install --extra-index https://test.pypi.org/simple/ doombuoy
```

## Usage



### Binary Classification

```python
import pandas as pd
from doombuoy import binary_classification_model_comparison_summary, binary_classification_model_comparison_models

# Prepare your DataFrame `df` with features and a binary target column, e.g., 'target'
summary = binary_classification_model_comparison_summary(df, target_col='target')
print(summary)

# To see model abbreviations and their full names:
binary_classification_model_comparison_models()
```

### Multiclass Classification

```python
from doombuoy import multiclass_classification_model_comparison_summary, multiclass_classification_model_comparison_models

# Prepare your DataFrame `df` with features and a multiclass target column, e.g., 'target'
summary = multiclass_classification_model_comparison_summary(df, target_col='target')
print(summary)

# To see model abbreviations and their full names:
multiclass_classification_model_comparison_models()
```

### Regression

```python
from doombuoy import regression_model_comparison_summary, regression_model_comparison_models

# Prepare your DataFrame `df` with features and a numeric target column, e.g., 'target'
summary = regression_model_comparison_summary(df, target_col='target')
print(summary)

# To see model abbreviations and their full names:
regression_model_comparison_models()
```

### Time Series Forecasting

```python
from doombuoy import time_series_forecasting_summary

# Prepare your DataFrame `df` with a time column (e.g., 'ds') and a target column (e.g., 'y')
summary = time_series_forecasting_summary(df, target_col='y', time_col='ds')
print(summary)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`doombuoy` was created by Agam Singh Saini. It is licensed under the terms of the Apache License 2.0 license.

## Credits

`doombuoy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
