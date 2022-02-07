from this import d
import numpy as np
import pandas as pd
from helpers.ml import log_transform, TargetEncoderCV
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import shap
import datetime
from fbprophet import Prophet
from helpers.ml import cyclic_encoding, prediction_fix
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger('ml_pipeline.'+__name__)
logging.basicConfig(filename='logs/errors.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')



class MLPipeline:
    def __init__(self, train_start_date, train_end_date,
                 test_start_date, test_end_date, prediction_start_date, prediction_end_date, target_col_name='sales'):

        assert prediction_end_date >= prediction_start_date > test_end_date >= test_start_date > train_end_date >= train_start_date
        if (prediction_start_date - test_end_date).days < 1:
            raise Exception(
                "Prediction interval must start immediately after validation period")

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.prediction_start_date = prediction_start_date
        self.prediction_end_date = prediction_end_date
        self.target_col_name = target_col_name
        self.predictions = {}

    def generate_train_columns(self, all_columns):
        except_cols = \
            ['date',
             'year_month',
             'year_quarter',
             'date_month',
             'weekday_week',
             'year_week',
             'target',
             'sales']
        train_cols = [c for c in all_columns if c not in except_cols]
        return train_cols

    def create_datasets(self, base_data):

        train_cols, target_col = self.generate_train_columns(
            base_data.columns), self.target_col_name

        train_data = base_data[base_data.date.between(self.train_start_date,
                                                      self.train_end_date)].reset_index(drop=True)
        test_data = base_data[base_data.date.between(self.test_start_date,
                                                     self.test_end_date)].reset_index(drop=True)
        predict_data = base_data[base_data.date.between(self.prediction_start_date,
                                                        self.prediction_end_date)].reset_index(drop=True)

        if target_col not in predict_data:
            predict_data[target_col] = 0

        cat_cols = []
        for column in train_cols:
            if train_data[column].dtype == 'object':
                cat_cols.append(column)

        te_cv = None
        if len(cat_cols) > 0:
            print('Categorical columns for target encoding: {}'.format(len(cat_cols)))
            print(cat_cols)
            te_cv = TargetEncoderCV(KFold(n_splits=3))
            enc_cols = []
            for c in cat_cols:
                train_data[c + '_tenc'] = np.nan
                test_data[c + '_tenc'] = np.nan
                predict_data[c + '_tenc'] = np.nan
                enc_cols.append(c + '_tenc')
            train_cols = [
                c + '_tenc' if c in cat_cols else c for c in train_cols]
            train_data[enc_cols] = te_cv.fit_transform(
                train_data[cat_cols], train_data[target_col])
            test_data[enc_cols] = te_cv.transform(test_data[cat_cols])
            predict_data[enc_cols] = te_cv.transform(predict_data[cat_cols])

        return (train_data[train_cols], log_transform(train_data[target_col])), (test_data[train_cols], log_transform(test_data[target_col])), \
            (predict_data, log_transform(predict_data[target_col]))

    def train_model(self, train_data, test_data):
        params = {
            'max_bin': [128],
            'num_leaves': [8],
            'reg_alpha': [1.2],
            'reg_lambda': [1.2],
            'min_data_in_leaf': [50],
            'learning_rate': [0.001]
        }
        print('Train data shape: NROWS = {0}, NCOLUMNS = {1}'.format(
            train_data[0].shape[0], train_data[0].shape[1]))
        print('Validation data shape: NROWS = {0}, NCOLUMNS = {1}'.format(
            test_data[0].shape[0], test_data[0].shape[1]))
        model = LGBMRegressor(n_estimators=10**5, n_jobs=-1, **params)
        model.fit(train_data[0], train_data[1],
                  eval_set=test_data,
                  eval_metric='rmse', verbose=1000, early_stopping_rounds=50)
        self.clf = model
        logger.info('Model trained')

    def show_feature_importance(self, data):
        explainer = shap.TreeExplainer(self.clf)
        shap.plots.beeswarm(
            explainer(data[self.clf.feature_name_]),
            max_display=25
        )

    @staticmethod
    def update_recursive_features(data, prediction_date):
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 28]:
            data['total_sales_last_%s_days' % str(lag)] = data['total_sales_last_%s_days' % str(
                lag)] - data['total_sales_last_1_days'] + data['predicted']

        if prediction_date.isocalendar()[1] > data['week']:
            data['week_sales_to_date'] = 0
        else:
            data['week_sales_to_date'] = data['predicted']

        if prediction_date.month > data['month']:
            data['month_sales_to_date'] = 0
        else:
            data['month_sales_to_date'] = data['predicted']

        data['percentage_sold_of_month'] = data['month_sales_to_date'] / \
            data['avg_same_month_sales']
        data['percentage_sold_of_week'] = data['week_sales_to_date'] / \
            data['avg_same_week_sales']

        data['date'] = prediction_date

        data['weekday'] = data.date.dt.dayofweek
        data['week'] = data.date.dt.isocalendar().week
        data['month'] = data.date.dt.month
        data['quarter'] = data.date.dt.quarter
        data['year'] = data.date.dt.year
        data = cyclic_encoding(data)

        return data

    def recursive_prediction(self, prediction_data):
        print('prediction cols', prediction_data.columns)
        data = prediction_data[0].copy()
        data = data[data['date'] == self.prediction_start_date]
        data['predicted'] = self.clf.predict(data[self.clf.feature_name_])
        range_of_dates = pd.date_range(self.prediction_start_date + datetime.timedelta(days=1),
                                       self.prediction_end_date, freq='d')
        for date in range_of_dates:
            new_prediction_data = self.update_recursive_features(
                data.tail(1).copy(), date)
            new_prediction_data['predicted'] = self.clf.predict(
                new_prediction_data[data[self.clf.feature_name_]])
            data = data.append(new_prediction_data[data.columns])
            print(f'Recurisely predicted date: str({date})')
        data['predicted'] = prediction_fix(log_transform(data['predicted'], mode='back'))
        logger.info('Recursive prediction finished')
        return data.reset_index(drop=True)

    @staticmethod
    def baseline_fbprophet(tr, pr):
        prophet = Prophet(daily_seasonality=True, weekly_seasonality=True,
                          yearly_seasonality=False, changepoint_prior_scale=0.01)
        prophet.fit(
            tr.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y']])

        preds = prophet.predict(
            pr.rename(columns={'date': 'ds'}))[['ds', 'yhat']]

        return prediction_fix(preds['yhat'].values)

    @staticmethod
    def baseline_average(tr, pr):
        # Based on seasonal average
        avg = tr.groupby(['weekday_week'])['sales'].mean().to_dict()
        if tr[tr['year'] == pr['year'].max()-1].shape[0] > 0:
            trend = tr[tr['year'] == pr['year'].max()]['sales'].sum().values / \
                tr[tr['year'] == pr['year'].max() - 1]['sales'].sum().values
        else:
            trend = np.ones(pr['weekday_week'].shape[0])

        return prediction_fix(pr['weekday_week'].map(avg).values*trend)

    def baseline_models(self, base_data):
        train = base_data[base_data.date.between(
            self.train_start_date, self.test_end_date)]
        prediction = base_data[base_data.date.between(
            self.prediction_start_date, self.prediction_end_date)]

        prediction_results = prediction[[
            'date', 'year', 'weekday_week', 'hierarchy1_id', 'storetype_id']].drop_duplicates()
        prediction_pairs = prediction_results[[
            'hierarchy1_id', 'storetype_id']].drop_duplicates()

        prediction_results['predicted_fbprophet'] = 0
        prediction_results['predicted_seasonal_average'] = 0

        for (hier, store) in prediction_pairs.to_numpy():
            train_data = train[(train.hierarchy1_id == hier) &
                               (train.storetype_id == store)]
            pred_data = prediction_results[(prediction_results.hierarchy1_id == hier) &
                                           (prediction_results.storetype_id == store)]

            prediction_results.loc[(prediction_results.hierarchy1_id == hier) &
                                   (prediction_results.storetype_id == store), 'predicted_fbprophet'] = \
                self.baseline_fbprophet(train_data, pred_data)

            prediction_results.loc[(prediction_results.hierarchy1_id == hier) &
                                   (prediction_results.storetype_id == store), 'predicted_seasonal_average'] = \
                self.baseline_average(train_data, pred_data)
            print(
                f'Making baseline predictions for product category {hier} and store type {store}')

        self.predictions['baseline'] = prediction_results[[
            'date', 'hierarchy1_id', 'storetype_id', 'predicted_fbprophet', 'predicted_fbprophet']]

    def combine_results(self):
        results = pd.merge(self.predictions['lgb'].rename(columns={'predicted': 'lgb_predicted'}),
                           self.predictions['baseline'], on=[
                               'date', 'hierarchy1_id', 'storetype_id']
                           )
        self.predictions['merged'] = results

    def lgb_model_building(self, base_data):

        print('Started building LightGBM model...')

        tr, vl, pr = self.create_datasets(base_data)

        self.train_model(tr, vl)
        print('LGB model trained!')

        predicted_data = self.recursive_prediction(pr)

        print('LGB predictions finished!')
        self.predictions['lgb'] = predicted_data[[
            'date', 'hierarchy1_id', 'storetype_id', 'sales', 'predicted']]

    def ml_pipeline(self, base_data):
        self.lgb_model_building(base_data)
        logger.info('LGB odel finished')
        self.baseline_models(base_data)
        logger.info('Baseline models finished')
        self.combine_results()
