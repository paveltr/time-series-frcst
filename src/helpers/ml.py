from numpy import sin, cos, pi, mean, log, exp, zeros_like, float64, ones, array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv
from category_encoders import CatBoostEncoder
import pandas as pd
from sklearn.model_selection import KFold


def custom_RMSE(y_true, y_pred):
    # Higher penalty for predicting lower values than greater ones
    delta = [1.1*(yp - yt) if yp < yt else 0.9*(yp - yt)
             for yt, yp in zip(y_true, y_pred)]
    squared_residual = array(delta)**2
    grad = squared_residual
    hess = ones(len(y_true))
    return 'wrmse', mean(grad), False


def prediction_fix(X):
    return [0 if x < 0 else x for x in X]


def cyclic_encoding(df, col, max_val_mode='auto'):
    if max_val_mode == 'auto':
        max_val = df[col].max()
    else:
        if col == 'week':
            max_val = 52
        elif col == 'month':
            max_val = 12
        elif col == 'quarter':
            max_val = 4
        elif col == 'weekday':
            max_val = 7

    df[col + '_sin'] = sin(2 * pi * df[col].astype(float)/max_val)
    df[col + '_cos'] = cos(2 * pi * df[col].astype(float)/max_val)
    if col not in ['week', 'month', 'weekday']:
        return df.drop([col], axis=1)
    else:
        return df


def MAPE(Y_actual, Y_Predicted):
    mape = mean(abs((Y_actual - Y_Predicted)/(Y_actual+1)))*100
    return mape


def log_transform(series, mode='forward'):
    if mode == 'forward':
        return log(series + 1)
    elif mode == 'back':
        return exp(series) - 1
    else:
        raise Exception(
            'The mode argument should be one of ("forward", "back")')


class TargetEncoderCV(BaseEstimator, TransformerMixin):
    '''
    Fold-based target encoder robust to overfitting
    '''

    def __init__(self, cv, **cbe_params):
        self.cv = cv
        self.cbe_params = cbe_params

    @property
    def _n_splits(self):
        return check_cv(self.cv).n_splits

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        self.cbe_ = []
        cv = check_cv(self.cv)
        cbe = CatBoostEncoder(
            cols=X.columns.tolist(),
            return_df=False,
            **self.cbe_params)

        X_transformed = zeros_like(X, dtype=float64)
        for train_idx, valid_idx in cv.split(X, y):
            self.cbe_.append(clone(cbe).fit(X.loc[train_idx], y[train_idx]))
            X_transformed[valid_idx] = self.cbe_[
                -1].transform(X.loc[valid_idx])
        return pd.DataFrame(X_transformed, columns=X.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = zeros_like(X, dtype=float64)
        for cbe in self.cbe_:
            X_transformed += cbe.transform(X) / self._n_splits
        return pd.DataFrame(X_transformed, columns=X.columns)
