from model_training import MLPipeline
from feature_engineering import FeaturePipeline
import yaml
from helpers.utils import read_csv, convert_to_date
import pandas as pd
import gc
import argparse
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('ml_pipeline.'+__name__)
logging.basicConfig(filename='logs/errors.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')


def update_dates(cfg):
    '''Converts string to datetime'''
    for key, value in cfg['MODEL'].items():
        if 'date' in key:
            cfg['MODEL'][key] = convert_to_date(value)
    return cfg


def forecasting_pipeine(data, cfg, output_predictions=True):
    '''This function creates features, trains a model and saves a file with forecasted values'''

    FeatureBuilder = FeaturePipeline(**cfg['MODEL'])
    FeatureBuilder.create_features(data)

    logger.info('Features readt')

    MLBuilder = MLPipeline(**cfg['MODEL'])
    MLBuilder.ml_pipeline(FeatureBuilder.features)

    logger.info('Predictions ready')

    if output_predictions:
        save_predictions(
            MLBuilder.predictions['merged'], cfg['OUTPUT']['path'])
    return MLBuilder.predictions['merged']


def save_predictions(predictions, path):
    df = predictions[['date', 'hierarchy1_id',
                      'storetype_id', 'sales', 'lgb_predicted']]
    df.to_csv(path, index=False)


def set_config(filepath):
    '''This function reads configuration file'''
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return update_dates(cfg)


def create_data(cfg):
    '''This function reads files and merges into one dataframe'''

    logger.info('Making data....')
    datasets = {}
    for key, value in cfg['INPUT'].items():
        datasets[key] = read_csv(value)

    data = pd.merge(datasets['sales'], datasets['taxonomy'], how='inner', on=['product_id'])\
        .merge(datasets['stores'], how='inner', on=['store_id'])

    del datasets
    gc.collect()
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ml_pipeline')
    parser.add_argument('--config_path', action="store",
                        dest='config_path', default='../prediction_config.yaml')

    # read ML config
    args = parser.parse_args()
    cfg = set_config(args.config_path)

    # read csv data
    data = create_data(cfg)

    # make predictions
    forecasting_pipeine(data, cfg)
