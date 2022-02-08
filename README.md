# Time series sales forecasting

This repository containts python script for time series forecasting of items sold.

# Python script usage

To make predictions, you need to put files with data to `data` folder. The script requires 3 files:
> sales.csv

> taxonomy.csv

> store.csv

To build and run docker image
```
bash make_predictions.sh
```

To create predictions inside of the `docker container`:

```
bash run_pipeline.sh
```

# Jupyter notebooks

There are two notebooks
> /notebooks/EDA.ipynb - exploratory data analysis

> /src/ML Evaluation.ipynb - shows analytics for model performance

## Model performance

Model performace is based on Weighted MAPE metric
![alt text](src/performance.png)