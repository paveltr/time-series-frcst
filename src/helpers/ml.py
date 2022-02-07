import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv
from category_encoders import CatBoostEncoder
import pandas as pd
from sklearn.model_selection import KFold

def prediction_fix(X):
    return [0 if x < 0 else x for x in X]

def cyclic_encoding(df, col):

    max_val = df[col].max()
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)

    return df.drop([col], axis=1)


def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/(Y_actual+1)))*100
    return mape


def log_transform(series, mode='forward'):
    if mode == 'forward':
        return np.log(series + 1)
    elif mode == 'back':
        return np.exp(series) - 1
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

        X_transformed = np.zeros_like(X, dtype=np.float64)
        for train_idx, valid_idx in cv.split(X, y):
            self.cbe_.append(clone(cbe).fit(X.loc[train_idx], y[train_idx]))
            X_transformed[valid_idx] = self.cbe_[
                -1].transform(X.loc[valid_idx])
        return pd.DataFrame(X_transformed, columns=X.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = np.zeros_like(X, dtype=np.float64)
        for cbe in self.cbe_:
            X_transformed += cbe.transform(X) / self._n_splits
        return pd.DataFrame(X_transformed, columns=X.columns)
