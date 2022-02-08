from this import d
import numpy as np
import pandas as pd
from helpers.ml import log_transform, TargetEncoderCV, cyclic_encoding, prediction_fix, custom_RMSE
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import shap
import datetime
from fbprophet import Prophet
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger('ml_pipeline.'+__name__)
logging.basicConfig(filename='logs/errors.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')


class MLPipeline:
    def __init__(self, train_start_date=None, train_end_date=None,
                 test_start_date=None, test_end_date=None, prediction_start_date=None, prediction_end_date=None,
                 target_col_name='sales', except_cols=['date',
                                                       'year_month',
                                                       'year_quarter',
                                                       'date_month',
                                                       'weekday_week',
                                                       'year_week',
                                                       'target',
                                                       'sales',
                                                       'week',
                                                       'month']):

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
        self.number_of_predictions = (
            self.prediction_end_date - self.prediction_start_date).days + 1
        self.except_cols = except_cols
        self.prediction_month = int(prediction_start_date.month)

    def create_weights(self, data):
        year_weights = {'2017': 0.5, '2018': 0.75, '2019': 1.0}
        data['weight'] = data['year'].map(year_weights)
        data.loc[data['weight'].isnull(), 'weight'] = data[data['weight'].isnull(
        )]['year'].map(lambda x: 0.25 if x < 2017 else 1.0)
        monthly_weights = {}
        for i in range(12):
            monthly_weights[i +
                            1] = 0.25 if (self.prediction_month - i) < 3 else 0

        data['weight'] = (
            data['weight'] + data['month'].map(monthly_weights)).fillna(1)
        return data

    def generate_train_columns(self, all_columns):

        train_cols = [c for c in all_columns if c not in self.except_cols]
        return train_cols

    def create_datasets(self, base_data):

        train_cols, target_col = self.generate_train_columns(
            base_data.columns), self.target_col_name

        base_data = self.add_fb_features(base_data)
        train_data = base_data[base_data.date.between(self.train_start_date,
                                                      self.train_end_date)].reset_index(drop=True)

        train_data = self.create_weights(train_data)
        test_data = base_data[base_data.date.between(self.test_start_date,
                                                     self.test_end_date)].reset_index(drop=True)
        predict_data = base_data[base_data.date.between(self.prediction_start_date,
                                                        self.prediction_end_date)].reset_index(drop=True)
        self.number_of_groups = predict_data[[
            'hierarchy1_id', 'storetype_id']].drop_duplicates().shape[0]

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
            (predict_data, log_transform(
                predict_data[target_col])), train_data['weight']

    def train_model(self, train_data, test_data, sample_weight=None):
        params = {
            'max_bin': [128],
            'max_depth': [5],
            'num_leaves': [50],
            'reg_alpha': [0.5],
            'reg_lambda': [0.5],
            'min_data_in_leaf': [50],
            'learning_rate': [0.01]
        }
        print('Train data shape: NROWS = {0}, NCOLUMNS = {1}'.format(
            train_data[0].shape[0], train_data[0].shape[1]))
        print('Validation data shape: NROWS = {0}, NCOLUMNS = {1}'.format(
            test_data[0].shape[0], test_data[0].shape[1]))
        model = LGBMRegressor(n_estimators=10**5, n_jobs=-1, **params)
        model.fit(train_data[0], train_data[1],
                  eval_set=test_data,
                  sample_weight=sample_weight,
                  eval_metric=custom_RMSE,
                  verbose=1000,
                  early_stopping_rounds=50)
        self.clf = model
        logger.info('Model trained')

    def show_feature_importance(self, data):
        explainer = shap.TreeExplainer(self.clf)
        shap.plots.beeswarm(
            explainer(data[self.clf.feature_name_]),
            max_display=25
        )

    def update_recursive_features(self, data, prediction_date):
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 28]:
            data['total_sales_last_%s_days' % str(lag)] = data['total_sales_last_%s_days' % str(
                lag)] - data['total_sales_last_1_days'] + prediction_fix(log_transform(data['predicted'], mode='back'))

        if int(prediction_date.isocalendar()[1]) > int(data['week'].values[0]):
            data['week_sales_to_date'] = 0
        else:
            data['week_sales_to_date'] = prediction_fix(
                log_transform(data['predicted'].values, mode='back'))

        if int(prediction_date.month) > int(data['month'].values[0]):
            data['month_sales_to_date'] = 0
        else:
            data['month_sales_to_date'] = prediction_fix(
                log_transform(data['predicted'].values, mode='back'))

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
        for col in ['weekday', 'week', 'month', 'quarter']:
            data = cyclic_encoding(data, col, max_val_mode='manual')
        data = self.add_fb_features(data)
        return data

    def recursive_prediction(self, prediction_data):
        data = prediction_data.copy()
        data = data[data['date'] == self.prediction_start_date]
        data['predicted'] = self.clf.predict(data[self.clf.feature_name_])
        range_of_dates = pd.date_range(self.prediction_start_date + datetime.timedelta(days=1),
                                       self.prediction_end_date, freq='d')
        for date in range_of_dates:
            new_prediction_data = self.update_recursive_features(
                data.tail(1).copy(), date)
            new_prediction_data['predicted'] = self.clf.predict(
                new_prediction_data[self.clf.feature_name_])
            data = data.append(new_prediction_data[data.columns])
            print(f'Recurisely predicted date: str({date})')
        data['predicted'] = prediction_fix(
            log_transform(data['predicted'], mode='back'))
        logger.info('Recursive prediction finished')

        if data.shape[0] != self.number_of_predictions:
            raise Exception(
                '''
                Number of LGB recursive prediction doesn't equal to number of days. 
                Expected: {0}, received: {1}'''.format(self.number_of_predictions,
                                                       data.shape[0]))
        return data.reset_index(drop=True)

    def lgb_prediction(self, pr_data):
        prediction_data = pr_data[0]
        prediction_pairs = prediction_data[[
            'hierarchy1_id', 'storetype_id']].drop_duplicates()
        pred_df = pd.DataFrame()
        for (hier, store) in prediction_pairs.to_numpy():
            preds = prediction_data[(prediction_data.hierarchy1_id == hier) &
                                    (prediction_data.storetype_id == store)]\
                .sort_values(by='date').head(1)

            pred_df = pred_df.append(
                self.recursive_prediction(preds), ignore_index=True)
        if pred_df.shape[0] != self.number_of_predictions*self.number_of_groups:
            raise Exception(
                '''
                Total of recursive predictions doesn't equal to number of days multiplied by number of groups. 
                Expected: {0}, received: {1}'''.format(self.number_of_predictions*self.number_of_groups,
                                                       pred_df.shape[0]))

        pred_df['sales'] = log_transform(pr_data[1].values, mode='back')
        return pred_df

    def create_fb_features(self, data):

        train = data[(data.date.between(self.train_start_date, self.train_end_date))][['date', 'hierarchy1_id',
                                                                                       'storetype_id', 'sales']]\
            .rename(columns={'date': 'ds', 'sales': 'y'})
        predict = data[(data.date.between(self.test_start_date, self.prediction_end_date))][['date', 'hierarchy1_id',
                                                                                             'storetype_id', 'sales']]\
            .rename(columns={'date': 'ds'})

        train['predicted'] = 0
        predict['predicted'] = 0

        prediction_pairs = data[['hierarchy1_id',
                                 'storetype_id']].drop_duplicates()

        for (hier, store) in prediction_pairs.to_numpy():
            tr = train[(train.hierarchy1_id == hier) &
                       (train.storetype_id == store)]
            pr = predict[(predict.hierarchy1_id == hier) &
                         (predict.storetype_id == store)]
            prophet = Prophet(daily_seasonality=True, weekly_seasonality=True,
                              yearly_seasonality=False, changepoint_prior_scale=0.01)
            prophet.fit(tr[['ds', 'y']])

            train.loc[(train.hierarchy1_id == hier) & (train.storetype_id == store), 'predicted'
                      ] = prophet.predict(tr[['ds']])
            predict.loc[(predict.hierarchy1_id == hier) & (
                predict.storetype_id == store), 'predicted'] = prophet.predict(pr[['ds']])

        dict_df = pd.concat([train[['ds', 'hierarchy1_id', 'storetype_id', 'predicted']],
                             predict[['ds', 'hierarchy1_id', 'storetype_id', 'predicted']]])
        dict_df['ds'] = dict_df['ds'].astype(str).str[:10]
        self.fb_encoder = dict_df.set_index(['ds', 'hierarchy1_id', 'storetype_id'])[
            'predicted'].to_dict()

    def add_fb_features(self, df):
        df['fb_feature'] = df[['date', 'hierarchy1_id', 'storetype_id']]\
            .apply(lambda x: self.fb_encoder[(str(x['date'])[:10], x['hierarchy1_id'], x['storetype_id'])], axis=1)
        return df

    @staticmethod
    def baseline_fbprophet(tr, pr, return_train=False):
        prophet = Prophet(daily_seasonality=True, weekly_seasonality=True,
                          yearly_seasonality=False, changepoint_prior_scale=0.01)
        prophet.fit(
            tr.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y']])

        preds = prophet.predict(
            pr.rename(columns={'date': 'ds'}))['yhat'].values
        if return_train:
            tr_preds = prophet.predict(tr.rename(columns={'date': 'ds'})[['ds']])[
                'yhat'].values
            return prediction_fix(tr_preds), prediction_fix(preds)
        return prediction_fix(preds)

    @staticmethod
    def baseline_average(tr, pr):
        # Based on seasonal average
        trend = 1
        if tr[tr['year'] < pr['year'].max()].shape[0] > 0:
            print('Taking average by weekday and week')
            avg = tr.groupby(['weekday_week'])['sales'].mean()
            if tr[tr['year'] == pr['year'].max()-1].shape[0] > 0:
                trend = tr[tr['year'] == pr['year'].max()]['sales'].sum() / \
                    tr[tr['year'] == pr['year'].max() - 1]['sales'].sum()

            print('Average sales: min={0:0.1f}, mean={1:0.1f}, max={1:0.1f}'.format(
                avg.min(), avg.mean(), avg.max()))
            return prediction_fix(pr['weekday_week'].map(avg.to_dict()).values*trend)

        else:
            print('Taking average by weekday only')
            avg = tr[tr.year == pr['year'].max()].groupby(['weekday'])[
                'sales'].mean()
            print('Average sales: min={0:0.1f}, mean={1:0.1f}, max={1:0.1f}'.format(
                avg.min(), avg.mean(), avg.max()))
            return prediction_fix(pr['weekday'].map(avg.to_dict()))

    def baseline_models(self, base_data):
        train = base_data[base_data.date.between(
            self.train_start_date, self.test_end_date)]
        prediction = base_data[base_data.date.between(
            self.prediction_start_date, self.prediction_end_date)]

        prediction_results = prediction[[
            'date', 'year', 'weekday_week', 'weekday', 'hierarchy1_id', 'storetype_id']].drop_duplicates()
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
            'date', 'hierarchy1_id', 'storetype_id', 'predicted_seasonal_average', 'predicted_fbprophet']]

        if self.number_of_predictions*self.number_of_groups != self.predictions['baseline'].shape[0]:
            raise Exception(
                '''
                    Total of baseline predictions doesn't equal to number of days multiplied by number of groups. 
                    Expected: {0}, received: {1}'''.format(self.number_of_predictions*self.number_of_groups,
                                                           self.predictions['baseline'].shape[0]))

    def combine_results(self):
        results = pd.merge(self.predictions['lgb'].rename(columns={'predicted': 'lgb_predicted'}),
                           self.predictions['baseline'], on=[
                               'date', 'hierarchy1_id', 'storetype_id']
                           )
        results['lgb_predicted'] = 0.5*results['lgb_predicted'] + 0.5*results['predicted_fbprophet']
        if results.shape[0] != self.number_of_predictions*self.number_of_groups:
            raise Exception(
                '''
                Total of predictions doesn't equal to number of days multiplied by number of groups. 
                Expected: {0}, received: {1}'''.format(self.number_of_predictions*self.number_of_groups,
                                                       results.shape[0]))
        self.predictions['merged'] = results

    def lgb_model_building(self, base_data):

        print('Started building LightGBM model...')

        self.create_fb_features(base_data)
        tr, vl, pr, w = self.create_datasets(base_data)

        self.train_model(tr, vl, sample_weight=w)
        print('LGB model trained!')

        predicted_data = self.lgb_prediction(pr)

        print('LGB predictions finished!')
        self.predictions['lgb'] = predicted_data[[
            'date', 'hierarchy1_id', 'storetype_id', 'sales', 'predicted']]

    def ml_pipeline(self, base_data):
        self.lgb_model_building(base_data)
        logger.info('LGB odel finished')
        self.baseline_models(base_data)
        logger.info('Baseline models finished')
        self.combine_results()
