# read version from installed package
from importlib.metadata import version
__version__ = version("doombuoy")

from .data.sets import (
	binary_classification_model_comparison_summary,
	multiclass_classification_model_comparison_summary,
	regression_model_comparison_summary,
	time_series_forecasting_summary,
    binary_classification_model_comparison_models,
    multiclass_classification_model_comparison_models,
    regression_model_comparison_models
)