# 🚀 DoomBuoy

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-2025.0.0.25-green.svg)](https://test.pypi.org/project/doombuoy/)

**DoomBuoy** is a comprehensive Python package designed to streamline machine learning workflows and accelerate data science projects. It automates model comparison, performance evaluation, and data quality assessment across multiple task types.

## ✨ Key Features

- **🎯 Automated Model Comparison**: Test multiple algorithms simultaneously for classification, regression, and time series tasks
- **📊 Comprehensive Metrics**: Evaluate models using accuracy, AUC, F1-score, precision, recall, Kappa, MCC, and more
- **🔍 Data Quality Analysis**: Automated detection of missing values, outliers, duplicates, and data quality issues
- **📈 Built-in Visualization**: Create performance plots and data quality dashboards with minimal code
- **⚡ Fast Prototyping**: Quickly establish baselines and identify top-performing models
- **🤖 Wide Algorithm Support**: Includes scikit-learn, XGBoost, LightGBM, CatBoost, and time series models

## 📦 Installation

```bash
pip install --extra-index-url https://test.pypi.org/simple/ doombuoy
```

### Requirements

- Python 3.11+
- NumPy, Pandas, Scikit-learn
- XGBoost, LightGBM, CatBoost
- Statsmodels, Seaborn, Matplotlib

## 🚀 Quick Start

```python
import pandas as pd
from doombuoy import binary_classification_model_comparison_summary

# Load your data
df = pd.read_csv('your_data.csv')

# Compare models with one line of code
results = binary_classification_model_comparison_summary(df, target_col='target')
print(results)
```

## 📚 Usage Guide

### 1️⃣ Binary Classification

Compare 16+ classification algorithms automatically. Returns a comprehensive performance summary with metrics like accuracy, AUC, precision, recall, F1-score, Kappa, and MCC.

```python
from doombuoy import (
    binary_classification_model_comparison_summary,
    binary_classification_model_comparison_models
)

# Run comparison on your dataset
results = binary_classification_model_comparison_summary(df, target_col='target')
print(results)

# View all available models
binary_classification_model_comparison_models()
```

**Supported Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, KNN, Decision Tree, Naive Bayes, LDA, QDA, AdaBoost, Extra Trees, Ridge Classifier, and Baseline Dummy Classifier.

### 2️⃣ Multiclass Classification

Evaluate multiple algorithms for multiclass problems with macro-averaged metrics.

```python
from doombuoy import (
    multiclass_classification_model_comparison_summary,
    multiclass_classification_model_comparison_models
)

# Compare models for multiclass classification
results = multiclass_classification_model_comparison_summary(df, target_col='target')
print(results)

# View model abbreviations
multiclass_classification_model_comparison_models()
```

### 3️⃣ Regression

Test regression algorithms with metrics including R², MAE, MSE, RMSE, and MAPE.

```python
from doombuoy import (
    regression_model_comparison_summary,
    regression_model_comparison_models
)

# Run regression model comparison
results = regression_model_comparison_summary(df, target_col='price')
print(results)

# View available regression models
regression_model_comparison_models()
```

**Supported Models**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVR, KNN, Decision Tree, and more.

### 4️⃣ Time Series Forecasting

Evaluate time series models with specialized metrics: MASE, RMSSE, SMAPE, and MAPE.

```python
from doombuoy import time_series_forecasting_summary

# Compare time series forecasting models
results = time_series_forecasting_summary(
    df, 
    target_col='sales',  # Target variable
    time_col='date'      # Time column
)
print(results)
```

**Supported Models**: ARIMA, SARIMA, Exponential Smoothing, Holt-Winters, Prophet, and statistical baselines.

### 5️⃣ Model Performance Visualization

Create beautiful visualizations to compare model performance across metrics.

```python
from doombuoy import plot_model_performance

# Get comparison results
results = binary_classification_model_comparison_summary(df, target_col='target')

# Create performance plots for different metrics
plot_model_performance(results, metric="Accuracy")
plot_model_performance(results, metric="AUC")
plot_model_performance(results, metric="F1 Score")
plot_model_performance(results, metric="Precision")
```

### 6️⃣ Data Quality Assessment

Automatically analyze data quality and identify potential issues before modeling.

```python
from doombuoy import (
    data_quality_report,
    data_quality_summary,
    plot_data_quality_overview,
    detect_potential_issues
)

# Comprehensive data quality analysis
quality_report = data_quality_report(df)
print(quality_report)

# High-level summary statistics
summary = data_quality_summary(df)
print(summary)

# Visual data quality dashboard
plot_data_quality_overview(df)

# Detect anomalies and potential issues
issues = detect_potential_issues(df)
print("Issues Detected:", issues)
```

**Features**: Missing value analysis, duplicate detection, outlier identification, data type validation, cardinality assessment, and statistical summaries.

## 📊 Example Output

### Model Comparison Results

```
Model       Accuracy    AUC     Recall  Precision  F1 Score  Kappa   MCC     TT (Sec)
------      --------    ---     ------  ---------  --------  -----   ---     --------
xgboost     0.9234     0.9567   0.9123   0.9345    0.9233   0.8456  0.8467   0.234
lightgbm    0.9198     0.9534   0.9087   0.9312    0.9198   0.8389  0.8401   0.187
rf          0.9156     0.9489   0.9045   0.9267    0.9155   0.8301  0.8314   0.456
catboost    0.9145     0.9478   0.9034   0.9256    0.9144   0.8278  0.8292   0.312
...
```

## 🎯 Use Cases

- **Rapid Prototyping**: Quickly test multiple algorithms to find promising approaches
- **Baseline Establishment**: Set performance benchmarks for your ML projects
- **Model Selection**: Compare models objectively across multiple metrics
- **Data Validation**: Identify data quality issues before training
- **Educational**: Learn about different algorithms and their performance characteristics
- **Production Ready**: Generate reproducible results with consistent evaluation protocols

## 🛠️ Advanced Features

### Model Abbreviations Reference

Each model comparison function includes a helper to display model abbreviations:

```python
# View all binary classification models
binary_classification_model_comparison_models()

# View all regression models
regression_model_comparison_models()

# View all multiclass models
multiclass_classification_model_comparison_models()
```

### Custom Train-Test Splits

All functions use stratified 80-20 train-test splits with `random_state=42` for reproducibility.

### Performance Metrics

- **Classification**: Accuracy, AUC-ROC, Precision, Recall, F1-Score, Cohen's Kappa, Matthews Correlation Coefficient
- **Regression**: R², MAE, MSE, RMSE, MAPE
- **Time Series**: MASE, RMSSE, SMAPE, MAPE
- **Training Time**: Seconds to fit each model

## 🤝 Contributing

Contributions are welcome! Please check out the [contributing guidelines](CONTRIBUTING.md) before getting started.

This project follows a [Code of Conduct](CONDUCT.md). By contributing, you agree to abide by its terms.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Doombuoyz/doombuoy.git
cd doombuoy

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## 📝 License

`doombuoy` is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Agam Singh Saini** - [GitHub](https://github.com/Doombuoyz)

## 🙏 Acknowledgments

- Built with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) using the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter)
- Powered by scikit-learn, XGBoost, LightGBM, CatBoost, and other amazing open-source libraries

## 📚 Documentation

For detailed documentation, examples, and tutorials, visit the [documentation site](docs/).

## 🐛 Issues & Support

Found a bug or have a feature request? Please open an issue on [GitHub Issues](https://github.com/Doombuoyz/doombuoy/issues).

## ⭐ Star History

If you find this package useful, please consider giving it a star on GitHub!

---

**Made with ❤️ for the Data Science community**
