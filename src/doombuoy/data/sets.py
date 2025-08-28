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

def binary_classification_model_comparison_summary(df, target_col):
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

def multiclass_classification_model_comparison_summary(df, target_col):
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

def regression_model_comparison_summary(df, target_col):
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

def time_series_forecasting_summary(df, target_col="y", time_col="ds", test_size=0.2):
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
