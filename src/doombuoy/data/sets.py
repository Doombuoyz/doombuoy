import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score,
    cohen_kappa_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def binary_classification_model_comparison_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    models = {
        "lr": LogisticRegression(max_iter=1000),
        "ridge": RidgeClassifier(),
        "lda": LinearDiscriminantAnalysis(),
        "rf": RandomForestClassifier(),
        "nb": GaussianNB(),
        "catboost": CatBoostClassifier(verbose=0) if CatBoostClassifier else None,
        "gbc": GradientBoostingClassifier(),
        "ada": AdaBoostClassifier(),
        "et": ExtraTreesClassifier(),
        "qda": QuadraticDiscriminantAnalysis(),
        "lightgbm": LGBMClassifier() if LGBMClassifier else None,
        "knn": KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss') if XGBClassifier else None,
        "dummy": DummyClassifier(strategy="most_frequent"),
        "svm": SVC(probability=True)
    }

    results = []
    for name, model in models.items():
        if model is None:
            continue
        start = time.time()
        model.fit(X_train, y_train)
        tt = time.time() - start
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4) if not np.isnan(auc) else None,
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1 Score": round(f1, 4),
            "Kappa": round(kappa, 4),
            "MCC": round(mcc, 4),
            "TT (Sec)": round(tt, 3)
        })

    return pd.DataFrame(results)

# Example usage:
# summary_df = binary_classification_model_comparison_summary(your_dataframe, 'target_column_name')
# print(summary_df)
#####################################################################################################################################################################
def binary_classification_model_comparison_models():
    model_map = {
    "lr": "LogisticRegression",
    "ridge": "RidgeClassifier",
    "lda": "LinearDiscriminantAnalysis",
    "rf": "RandomForestClassifier",
    "nb": "GaussianNB",
    "catboost": "CatBoostClassifier",
    "gbc": "GradientBoostingClassifier",
    "ada": "AdaBoostClassifier",
    "et": "ExtraTreesClassifier",
    "qda": "QuadraticDiscriminantAnalysis",
    "lightgbm": "LGBMClassifier",
    "knn": "KNeighborsClassifier",
    "dt": "DecisionTreeClassifier",
    "xgboost": "XGBClassifier",
    "dummy": "DummyClassifier",
    "svm": "SVC"
    }
    print(model_map)




####################################################################################################################################################################
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def multiclass_classification_model_comparison_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    models = {
        "lr": LogisticRegression(max_iter=1000, multi_class='auto'),
        "ridge": RidgeClassifier(),
        "lda": LinearDiscriminantAnalysis(),
        "rf": RandomForestClassifier(),
        "nb": GaussianNB(),
        "catboost": CatBoostClassifier(verbose=0) if CatBoostClassifier else None,
        "gbc": GradientBoostingClassifier(),
        "ada": AdaBoostClassifier(),
        "et": ExtraTreesClassifier(),
        "qda": QuadraticDiscriminantAnalysis(),
        "lightgbm": LGBMClassifier() if LGBMClassifier else None,
        "knn": KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') if XGBClassifier else None,
        "dummy": DummyClassifier(strategy="most_frequent"),
        "svm": SVC(probability=True, decision_function_shape='ovr')
    }

    results = []
    for name, model in models.items():
        if model is None:
            continue
        start = time.time()
        model.fit(X_train, y_train)
        tt = time.time() - start
        y_pred = model.predict(X_test)
        # For multiclass, use macro average for recall, precision, f1
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except Exception:
                auc = np.nan
        else:
            y_proba = None
            auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4) if not np.isnan(auc) else None,
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1 Score": round(f1, 4),
            "Kappa": round(kappa, 4),
            "MCC": round(mcc, 4),
            "TT (Sec)": round(tt, 3)
        })

    return pd.DataFrame(results)

# Example usage:
# summary_df = multiclass_classification_model_comparison_summary(your_dataframe, 'target_column_name')
# print(summary_df)

#########################################################################################################################################################
def multiclass_classification_model_comparison_models():
    multi_class_model_map = {
        "lr": "LogisticRegression",
        "ridge": "RidgeClassifier",
        "lda": "LinearDiscriminantAnalysis",
        "rf": "RandomForestClassifier",
        "nb": "GaussianNB",
        "catboost": "CatBoostClassifier",
        "gbc": "GradientBoostingClassifier",
        "ada": "AdaBoostClassifier",
        "et": "ExtraTreesClassifier",
        "qda": "QuadraticDiscriminantAnalysis",
        "lightgbm": "LGBMClassifier",
        "knn": "KNeighborsClassifier",
        "dt": "DecisionTreeClassifier",
        "xgboost": "XGBClassifier",
        "dummy": "DummyClassifier",
        "svm": "SVC"
        }
    print(multi_class_model_map)

############################################################################################################################################################################
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit,
    BayesianRidge, HuberRegressor, PassiveAggressiveRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

def regression_model_comparison_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    models = {
        "Dummy Regressor": DummyRegressor(),
        "omp": OrthogonalMatchingPursuit(),
        "en": ElasticNet(),
        "br": BayesianRidge(),
        "ridge": Ridge(),
        "llar": LassoLars(),
        "lasso": Lasso(),
        "huber": HuberRegressor(),
        "lar": Lars(),
        "gbrt": GradientBoostingRegressor(),
        "rf": RandomForestRegressor(),
        "catboost": CatBoostRegressor(verbose=0) if CatBoostRegressor else None,
        "par": PassiveAggressiveRegressor(max_iter=1000, tol=1e-3),
        "knn": KNeighborsRegressor(),
        "lightgbm": LGBMRegressor() if LGBMRegressor else None,
        "ada": AdaBoostRegressor(),
        "dt": DecisionTreeRegressor(),
        "et": ExtraTreesRegressor(),
    }

    results = []
    for name, model in models.items():
        if model is None:
            continue
        start = time.time()
        try:
            model.fit(X_train, y_train)
            tt = time.time() - start
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            # RMSLE and MAPE can fail if y or y_pred has negatives or zeros
            try:
                rmsle = mean_squared_log_error(y_test.clip(min=0), np.clip(y_pred, a_min=0, a_max=None)) ** 0.5
            except Exception:
                rmsle = np.nan
            try:
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)))
            except Exception:
                mape = np.nan
        except Exception:
            mae = rmse = r2 = rmsle = mape = np.nan
            tt = 0.0

        results.append({
            "Model": name,
            "MAE": round(mae, 4) if not np.isnan(mae) else None,
            "RMSE": round(rmse, 4) if not np.isnan(rmse) else None,
            "R2": round(r2, 4) if not np.isnan(r2) else None,
            "RMSLE": round(rmsle, 4) if not np.isnan(rmsle) else None,
            "MAPE": round(mape, 4) if not np.isnan(mape) else None,
            "TT (Sec)": round(tt, 2)
        })

    return pd.DataFrame(results)

# Example usage:
# summary_df = regression_model_comparison_summary(your_dataframe, 'target_column_name')
# print(summary_df)


##############################################################################################################################################################
def regression_model_comparison_models():
    regression_model_map = {
    "Dummy Regressor": "DummyRegressor",
    "omp": "OrthogonalMatchingPursuit",
    "en": "ElasticNet",
    "br": "BayesianRidge",
    "ridge": "Ridge",
    "llar": "LassoLars",
    "lasso": "Lasso",
    "huber": "HuberRegressor",
    "lar": "Lars",
    "gbrt": "GradientBoostingRegressor",
    "rf": "RandomForestRegressor",
    "catboost": "CatBoostRegressor",
    "par": "PassiveAggressiveRegressor",
    "knn": "KNeighborsRegressor",
    "lightgbm": "LGBMRegressor",
    "ada": "AdaBoostRegressor",
    "dt": "DecisionTreeRegressor",
    "et": "ExtraTreesRegressor"
        }
    print(regression_model_map)





######################################################################################################################################################
import time
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For classical/statistical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.theta import ThetaModel

# For naive and seasonal naive
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.deterministic import DeterministicProcess

# For ML models
from sklearn.linear_model import (
    LinearRegression, Ridge, ElasticNet, Lasso, Lars, LassoLars, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def mase(y_true, y_pred, y_train):
    n = y_train.shape[0]
    d = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d if d != 0 else np.nan

def rmsse(y_true, y_pred, y_train):
    n = y_train.shape[0]
    d = np.square(np.diff(y_train)).sum() / (n-1)
    errors = np.square(y_true - y_pred)
    return np.sqrt(errors.mean() / d) if d != 0 else np.nan

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))

def time_series_forecasting_summary(df: pd.DataFrame, target_col: str = "y", time_col: str = "ds", test_size: float = 0.2) -> pd.DataFrame:
    """
    Returns a summary DataFrame of time series forecasting models and metrics.
    Assumes df has columns [time_col, target_col].
    """
    df = df.sort_values(time_col)
    y = df[target_col].values
    n = len(y)
    n_train = int(n * (1 - test_size))
    y_train, y_test = y[:n_train], y[n_train:]
    index_train = df[time_col].iloc[:n_train]
    index_test = df[time_col].iloc[n_train:]

    results = []

    # ETS (Exponential Smoothing)
    try:
        start = time.time()
        ets = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
        y_pred = ets.forecast(len(y_test))
        tt = time.time() - start
        results.append({
            "Model": "ets",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Exponential Smoothing (Simple)
    try:
        start = time.time()
        exp_smooth = SimpleExpSmoothing(y_train).fit()
        y_pred = exp_smooth.forecast(len(y_test))
        tt = time.time() - start
        results.append({
            "Model": "exp_smooth",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # ARIMA
    try:
        start = time.time()
        arima = SARIMAX(y_train, order=(1,1,1)).fit(disp=False)
        y_pred = arima.forecast(len(y_test))
        tt = time.time() - start
        results.append({
            "Model": "arima",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Theta Forecaster
    try:
        start = time.time()
        theta = ThetaModel(y_train).fit()
        y_pred = theta.forecast(len(y_test))
        tt = time.time() - start
        results.append({
            "Model": "theta",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Naive Forecaster
    try:
        start = time.time()
        y_pred = np.repeat(y_train[-1], len(y_test))
        tt = time.time() - start
        results.append({
            "Model": "naive",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Seasonal Naive Forecaster (if seasonality is present)
    try:
        season_length = 12  # Change as appropriate for your data
        start = time.time()
        y_pred = np.array([y_train[-season_length + (i % season_length)] for i in range(len(y_test))])
        tt = time.time() - start
        results.append({
            "Model": "snaive",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Polynomial Trend Forecaster (degree=2)
    try:
        start = time.time()
        X_train = np.arange(n_train).reshape(-1, 1)
        X_test = np.arange(n_train, n).reshape(-1, 1)
        poly = np.polyfit(X_train.flatten(), y_train, 2)
        y_pred = np.polyval(poly, X_test.flatten())
        tt = time.time() - start
        results.append({
            "Model": "polytrend",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Grand Means Forecaster
    try:
        start = time.time()
        y_pred = np.repeat(np.mean(y_train), len(y_test))
        tt = time.time() - start
        results.append({
            "Model": "grand_means",
            "MASE": round(mase(y_test, y_pred, y_train), 4),
            "RMSSE": round(rmsse(y_test, y_pred, y_train), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 4),
            "MAPE": round(mape(y_test, y_pred)/100, 4),
            "SMAPE": round(smape(y_test, y_pred)/100, 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "TT (Sec)": round(tt, 4)
        })
    except Exception:
        pass

    # Add more models as needed (ML, ARIMA, etc.)
    # You can expand this section to include ML regressors with lag features if desired.

    return pd.DataFrame(results)

# Example usage:
# summary_df = time_series_forecasting_summary(your_dataframe, target_col="y", time_col="ds")
# print(summary_df)
#######################################################################################################################################################################

def plot_model_performance(results_df: pd.DataFrame, metric: str = "Accuracy"):
    """
    Visualize the performance of models using a bar plot and display values above the bars.

    Args:
        results_df (pd.DataFrame): DataFrame containing model performance metrics.
        metric (str): The metric to visualize (e.g., "Accuracy", "AUC", "F1 Score").

    Returns:
        None: Displays the plot.
    """
    results_df=results_df.sort_values(metric, ascending=False)
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results DataFrame.")
    
    # Adjust figure size based on the number of models
    num_models = len(results_df["Model"])
    fig_width = max(10, num_models * 1.5)  # Minimum width of 10, increase by 1.5 per model
    plt.figure(figsize=(fig_width, 6))
    
    # Create the bar plot
    ax = sns.barplot(data=results_df, x="Model", y=metric, palette="viridis")
    plt.title(f"Model Performance Comparison ({metric})", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add values above the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.4f}',  # Format the value to 4 decimal places
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.show()

    
#######################################################################################################################################################################

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive data quality report for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        pd.DataFrame: Data quality report with various metrics
    """
    
    report_data = []
    
    for column in df.columns:
        col_data = df[column]
        
        # Basic info
        total_count = len(col_data)
        non_null_count = col_data.count()
        null_count = col_data.isnull().sum()
        null_percentage = round((null_count / total_count) * 100, 2)
        
        # Data type
        dtype = str(col_data.dtype)
        
        # Duplicates
        duplicate_count = col_data.duplicated().sum()
        duplicate_percentage = round((duplicate_count / total_count) * 100, 2)
        
        # Unique values
        unique_count = col_data.nunique()
        unique_percentage = round((unique_count / total_count) * 100, 2)
        
        # Memory usage
        memory_usage = col_data.memory_usage(deep=True)
        
        # Initialize statistical metrics
        mean_val = median_val = std_val = min_val = max_val = q25 = q75 = skewness = kurtosis = None
        outlier_count = outlier_percentage = 0
        
        # Statistical analysis for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            mean_val = round(col_data.mean(), 4) if not col_data.empty else None
            median_val = round(col_data.median(), 4) if not col_data.empty else None
            std_val = round(col_data.std(), 4) if not col_data.empty else None
            min_val = round(col_data.min(), 4) if not col_data.empty else None
            max_val = round(col_data.max(), 4) if not col_data.empty else None
            q25 = round(col_data.quantile(0.25), 4) if not col_data.empty else None
            q75 = round(col_data.quantile(0.75), 4) if not col_data.empty else None
            
            # Skewness and Kurtosis
            try:
                skewness = round(col_data.skew(), 4) if not col_data.empty else None
                kurtosis = round(col_data.kurtosis(), 4) if not col_data.empty else None
            except:
                pass
            
            # Outlier detection using IQR method
            if q25 is not None and q75 is not None:
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = round((outlier_count / total_count) * 100, 2)
        
        # Most frequent value for categorical/text columns
        most_frequent = None
        most_frequent_count = 0
        if not col_data.empty:
            try:
                mode_series = col_data.mode()
                if not mode_series.empty:
                    most_frequent = str(mode_series.iloc[0])
                    most_frequent_count = (col_data == mode_series.iloc[0]).sum()
            except:
                pass
        
        # Zero values (for numeric columns)
        zero_count = zero_percentage = 0
        if pd.api.types.is_numeric_dtype(col_data):
            zero_count = (col_data == 0).sum()
            zero_percentage = round((zero_count / total_count) * 100, 2)
        
        # Infinity values (for numeric columns)
        inf_count = 0
        if pd.api.types.is_numeric_dtype(col_data):
            inf_count = np.isinf(col_data).sum()
        
        report_data.append({
            'Column': column,
            'Data_Type': dtype,
            'Total_Count': total_count,
            'Non_Null_Count': non_null_count,
            'Null_Count': null_count,
            'Null_Percentage': null_percentage,
            'Unique_Count': unique_count,
            'Unique_Percentage': unique_percentage,
            'Duplicate_Count': duplicate_count,
            'Duplicate_Percentage': duplicate_percentage,
            'Mean': mean_val,
            'Median': median_val,
            'Std_Dev': std_val,
            'Min': min_val,
            'Max': max_val,
            'Q25': q25,
            'Q75': q75,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Zero_Count': zero_count,
            'Zero_Percentage': zero_percentage,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_percentage,
            'Inf_Count': inf_count,
            'Most_Frequent_Value': most_frequent,
            'Most_Frequent_Count': most_frequent_count,
            'Memory_Usage_Bytes': memory_usage
        })
    
    return pd.DataFrame(report_data)

def data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Generate a high-level data quality summary.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        dict: Summary statistics about the DataFrame
    """
    
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    missing_percentage = round((total_missing / total_cells) * 100, 2)
    
    # Data type distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Memory usage
    total_memory = df.memory_usage(deep=True).sum()
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_rows_percentage = round((duplicate_rows / len(df)) * 100, 2)
    
    summary = {
        'Dataset_Shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'Total_Cells': total_cells,
        'Missing_Values': total_missing,
        'Missing_Percentage': missing_percentage,
        'Duplicate_Rows': duplicate_rows,
        'Duplicate_Rows_Percentage': duplicate_rows_percentage,
        'Numeric_Columns': len(numeric_cols),
        'Categorical_Columns': len(categorical_cols),
        'Datetime_Columns': len(datetime_cols),
        'Total_Memory_Usage_MB': round(total_memory / (1024 * 1024), 2),
        'Numeric_Column_Names': numeric_cols,
        'Categorical_Column_Names': categorical_cols,
        'Datetime_Column_Names': datetime_cols
    }
    
    return summary

def plot_data_quality_overview(df: pd.DataFrame):
    """
    Create visualizations for data quality overview.
    
    Args:
        df (pd.DataFrame): Input DataFrame to visualize
    """
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Quality Overview', fontsize=16)
    
    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        missing_data = df.isnull()
        sns.heatmap(missing_data, cbar=True, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Missing Values Heatmap')
        axes[0, 0].set_xlabel('Columns')
        axes[0, 0].set_ylabel('Rows')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Missing Values Heatmap')
    
    # 2. Missing values bar plot
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) > 0:
        missing_counts.plot(kind='bar', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title('Missing Values by Column')
        axes[0, 1].set_ylabel('Missing Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Missing Values by Column')
    
    # 3. Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Data Types Distribution')
    
    # 4. Unique values distribution
    unique_counts = df.nunique()
    axes[1, 1].hist(unique_counts.values, bins=20, color='lightblue', edgecolor='black')
    axes[1, 1].set_title('Distribution of Unique Values per Column')
    axes[1, 1].set_xlabel('Number of Unique Values')
    axes[1, 1].set_ylabel('Number of Columns')
    
    plt.tight_layout()
    plt.show()

# Example usage functions
def detect_potential_issues(df: pd.DataFrame) -> dict:
    """
    Detect potential data quality issues.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        dict: Dictionary of potential issues found
    """
    
    issues = {}
    
    # High missing values
    missing_threshold = 50  # 50% threshold
    high_missing_cols = []
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > missing_threshold:
            high_missing_cols.append((col, missing_pct))
    
    if high_missing_cols:
        issues['High_Missing_Values'] = high_missing_cols
    
    # Low variance columns (potential constant columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    low_variance_cols = []
    for col in numeric_cols:
        if df[col].var() < 0.01:  # Very low variance
            low_variance_cols.append(col)
    
    if low_variance_cols:
        issues['Low_Variance_Columns'] = low_variance_cols
    
    # High cardinality columns
    high_cardinality_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:  # More than 90% unique values
            high_cardinality_cols.append((col, df[col].nunique()))
    
    if high_cardinality_cols:
        issues['High_Cardinality_Columns'] = high_cardinality_cols
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues['Duplicate_Rows'] = duplicate_count
    
    return issues