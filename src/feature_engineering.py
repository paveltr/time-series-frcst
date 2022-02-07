import numpy as np
import pandas as pd
import datetime
from helpers.utils import add_month_to_year_month, generate_pivot_features
import logging

logger = logging.getLogger('ml_pipeline.'+__name__)
logging.basicConfig(filename='logs/errors.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')

class FeaturePipeline:
    def __init__(self, start_date, end_date):
        self.features = None
        self.start_date = start_date
        self.end_date = end_date
    
    @staticmethod
    def cyclic_encoding(df, col):
        
        max_val = df[col].max()
        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        
        return df.drop([col], axis=1)

    def add_date_columns(self, data):
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data = data[(data.date.between(self.start_date, self.end_date))]
        data['weekday'] = data.date.dt.dayofweek
        data['week'] = data.date.dt.isocalendar().week
        data['month'] = data.date.dt.month
        data['quarter'] = data.date.dt.quarter
        data['year'] = data.date.dt.year
        data['year_month'] = data.date.dt.strftime('%Y-%m')
        data['date_month'] = data.date.dt.strftime('%m-%d')
        data['year_quarter'] = pd.PeriodIndex(data.date, freq='Q')
        data['weekday_week'] = data['week'].astype(str) + '-' + data['weekday'].astype(str)
        data['year_week'] = data['year'].astype(str) + '-' + data['week'].astype(str)
        logger.info('Added columns with dates')
        return data

    @staticmethod
    def create_base_data(data):
        base_data = data.groupby(['date', 'hierarchy1_id', 'storetype_id', 'weekday', 'week', 'month',
                            'quarter', 'year', 'year_month', 'year_quarter', 'date_month',
                            'weekday_week', 'year_week'])['sales'].sum()\
                    .reset_index()\
                    .sort_values(by=['hierarchy1_id', 'storetype_id', 'date'])
        logger.info('Created basic data')
        return base_data

    @staticmethod
    def create_product_features(data):
        product_features = data.groupby(['hierarchy1_id', 'storetype_id', 'year_month']).agg({'product_id' : 'nunique',
                                                                                    'product_length' :  [np.min, np.mean, np.max],
                                                                                    'product_depth' : [np.min, np.mean, np.max],
                                                                                    'product_width' : [np.min, np.mean, np.max],
                                                                                    'store_id' : 'nunique',
                                                                                    'store_size' : [np.min, np.mean, np.max],
                                                                                    'price' : [np.min, np.mean, np.max]
                                                                                    }).reset_index()

        product_features.columns = ['hierarchy1_id', 'storetype_id', 'year_month', 'number_of_products', 
                                    'product_length_min', 'product_length_mean', 'product_length_max',
                                    'product_depth_min', 'product_depth_mean', 'product_depth_max',
                                    'product_width_min', 'product_width_mean', 'product_width_max', 'number_of_stores',
                                    'store_size_min', 'store_size_mean', 'store_size_max', 'price_min', 'price_mean', 'price_max']

        product_features['year_month'] = add_month_to_year_month(product_features['year_month'])
        logger.info('Added product features')
        return product_features


    @staticmethod
    def create_categorical_features(df, features=['promo_type_1', 'promo_bin_1', 'promo_type_2', 
                                            'promo_bin_2', 'promo_discount_2', 
                                            'promo_discount_type_2', 'city_id', 'cluster_id'],
                                index_level=['year_month', 'hierarchy1_id', 'storetype_id']):
        
        features_df = df[index_level].drop_duplicates().sort_values(by=index_level)
        for feature in features:
            print(f"Processing feature: {feature}")
            df[feature] = df[feature].astype(str).fillna('NaN')
            cat_features = generate_pivot_features(df, feature, index_level)
            cat_features.columns = [c if i < len(index_level) else str(c) + f'_{feature}' for i, c in enumerate(cat_features.columns)]
            features_df = pd.merge(features_df, cat_features, 
                                how='left', on=index_level)

        features_df['year_month'] = add_month_to_year_month(features_df['year_month'])
        logger.info('Added categorical features')
        return features_df.reset_index(drop=True).fillna(0)

    @staticmethod
    def create_lag_features(base_data, last_days = [1, 2, 3, 4, 5, 6, 7, 14, 28]):
        for lag in last_days:
            base_data['total_sales_last_%s_days' % str(lag)] = 0
            base_data['lag_%s' % str(lag)] = (base_data['date'] - base_data.groupby(['hierarchy1_id', 'storetype_id'])['date'].shift(lag)).dt.days
            base_data['lag_%s_sales' % str(lag)] = base_data.groupby(['hierarchy1_id', 'storetype_id'])['sales'].shift(lag)
            base_data['sales_last_%s_days' % str(lag)] = base_data['lag_%s_sales' % str(lag)]*(base_data['lag_%s' % str(lag)] == lag)
            base_data.drop(['lag_%s' % str(lag), 'lag_%s_sales' % str(lag)], axis=1, inplace=True)

        lag_columns = ['sales_last_%s_days' % str(lag) for lag in last_days]
        for i, lag in enumerate([1, 2, 3, 4, 5, 6, 7, 14, 28]):
            base_data['total_sales_last_%s_days' % str(lag)] = base_data[lag_columns[:lag]].sum(axis=1)
        base_data.drop(lag_columns, axis=1, inplace=True)
        logger.info('Added lag features')
        return base_data

    @staticmethod
    def create_hist_features(base_data):
        year_month_avg = base_data.groupby(['year', 'month', 'hierarchy1_id', 'storetype_id'])['sales'].sum().reset_index()
        year_month_avg.rename(columns={'sales' : 'avg_same_month_sales'}, inplace=True)
        
        year_week_avg = base_data.groupby(['year', 'week', 'hierarchy1_id', 'storetype_id'])['sales'].sum().reset_index()
        year_week_avg.rename(columns={'sales' : 'avg_same_week_sales'}, inplace=True)
        
        year_weekday_avg = base_data.groupby(['year', 'weekday_week', 'hierarchy1_id', 'storetype_id'])['sales'].sum().reset_index()
        year_weekday_avg.rename(columns={'sales' : 'avg_same_day_sales'}, inplace=True)
        
        year_month_avg['year'] = year_month_avg['year'] + 1
        year_week_avg['year'] = year_week_avg['year'] + 1
        year_weekday_avg['year'] = year_weekday_avg['year'] + 1
        
        base_data = pd.merge(base_data, year_month_avg, on=['year', 'month', 'hierarchy1_id', 'storetype_id'], how='left')
        base_data = pd.merge(base_data, year_week_avg, on=['year', 'week', 'hierarchy1_id', 'storetype_id'], how='left')
        base_data = pd.merge(base_data, year_weekday_avg, on=['year', 'weekday_week', 'hierarchy1_id', 'storetype_id'], how='left')
        logger.info('Added historical features')
        return base_data

    @staticmethod
    def create_cum_features(base_data):
        base_data['month_sales_to_date'] = base_data.groupby(['hierarchy1_id', 'storetype_id', 'year_month'])['sales'].cumsum() - base_data.sales
        base_data['week_sales_to_date'] = base_data.groupby(['hierarchy1_id', 'storetype_id', 'year_week'])['sales'].cumsum() - base_data.sales
        logger.info('Added cumulative features')
        return base_data

    @staticmethod
    def create_relative_features(base_data):
        base_data['percentage_sold_of_month'] = base_data['month_sales_to_date'] / base_data['avg_same_month_sales']
        base_data['percentage_sold_of_week'] = base_data['week_sales_to_date'] / base_data['avg_same_week_sales']
        logger.info('Added relative features')
        return base_data

    @staticmethod
    def time_feature_encoding(base_data):
        for period in ['weekday', 'week', 'month', 'quarter']:
            base_data = FeaturePipeline.cyclic_encoding(base_data, period)
        logger.info('Added time encoded features')
        return base_data 


    def create_features(self, data):
        logger.info('Feature building pipeline started...')
        data = self.add_date_columns(data)
        bData = self.create_base_data(data)
        # process features
        prodF = self.create_product_features(data)
        catF = self.create_categorical_features(data)
        bData = self.create_lag_features(bData)
        bData = self.create_hist_features(bData)
        bData = self.create_cum_features(bData)
        bData = self.create_relative_features(bData)
        bData = self.time_feature_encoding(bData)
        # add remaining features
        bData = pd.merge(bData, catF, how='left', on=['year_month', 'hierarchy1_id', 'storetype_id'])
        bData = pd.merge(bData, prodF, how='left', on=['year_month', 'hierarchy1_id', 'storetype_id'])
        logger.info('Feature building pipeline finished!')
        self.features = bData

        


