# Recommendation System in Electronic Commerce

![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![S3](https://img.shields.io/badge/S3-003366?style=for-the-badge)
![NetworkX](https://img.shields.io/badge/NetworkX-grey?style=for-the-badge&logo=NetworkX&logoColor=grey)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-219ebc?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-CC0066?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-yellow?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-778da9?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-778dc9?style=for-the-badge)
![implicit](https://img.shields.io/badge/implicit-000000?style=for-the-badge&logo=implicit&logoColor=white)


## Description

This part of the repository contains scripts and files about how to build the recommendation model.

The [dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) contains the following dataframes:
1. `category_tree` - a table with two columns: "parent category" and "child category". Typical way of representing the table as a tree.
2. `events` - table with the log of events:
    - `timestamp` - timestamp of the event,
    - `visitorid` - user ID,
    - `event` - event (`view`, `add to cart`, `transaction`),
    - `itemid` - item ID,
    - `transactionid` - transaction (purchase) ID.
3. `item_properties` - table with product properties:
    - `timestamp` - timestamp of adding property,
    - `itemid` - item ID,
    - `property` - property of the item,
    - `value` - property value


The task is to increase the number of `add to cart` events (i.e. adding an item to the cart by a user). 

Though there are other events apart from the `add to cart` event, it is possible to utilise the data about other events to build a recommendation model that recommends items that are more likely to be added to the cart:
- consider all events when training base models, but use different weights for different events. In this case, the `transaction` event has the highest weight, and the `view` event has the lowest weight.
- consider only at least `add to cart` events when making recommendations by all models and evaluating all models


## Project Structure

**[artifacts](/experiments/artifacts)**: This directory contains artifacts

**[config](/experiments/config)**: Configuration files directory
- [components.yaml](/experiments/config/components.yaml): Configuration for the project
- [logger_config.yaml](/experiments/config/logger_config.yaml): Configuration for the logger
- [dates.yaml](/experiments/config/dates.yaml): Configuration with the dates used to split the data
- [experiment_name.yaml](/experiments/config/experiment_name.yaml): Configuration with the experiment name that can change
- [airflow.yaml](/experiments/config/airflow.yaml): Configuration of the pipeline DAG.
- [tg_callback_template.yaml](/experiments/config/tg_callback_template.yaml): Template telegram callback configuration 

**[mlflow_server](/experiments/mlflow_server)**: This directory contains scripts and files related to the MLflow service
- [start.sh](/experiments/mlflow_server/start.sh): Entrypoint to start the MLflow server
- [clean.sh](/experiments/mlflow_server/clean.sh): Script to remove deleted data from the MLflow server
- [Dockerfile](/experiments/mlflow_server/Dockerfile): Dockerfile used to build the docker image for the MLFlow
- [requirements.txt](/experiments/mlflow_server/requirements.txt): List of required Python packages for the MLFlow service

**[notebooks](/experiments/notebooks)**: This directory jupyter notebooks where the experiments are performed
- [recommendations.ipynb](/experiments/notebooks/recommendations.ipynb): Notebook with the experiments

**[scripts](/experiments/scripts)**: This directory contains Python scripts used for conducting the experiments and for Airflow
- **[components](/experiments/scripts/components)**: A directory with the components that define Airflow pipeline

**[dags](/experiments/dags)**: This directory contains Python scripts that define Airflow DAGs
- [master_dag.py](/experiments/dags/master_dag.py): The DAG that defines the pipeline

[.env_template](/experiments/.env_template): This is a template file for the environment variables

[requirements.txt](/experiments/requirements.txt): List of required Python packages

[setup.py](/experiments/setup.py): Setup file for packaging python scripts, so that all scripts can be easily accessable from any directory

[Dockerfile](/experiments/Dockerfile): Dockerfile used to build the docker image for the Airflow

[docker-compose.yaml](/experiments/docker-compose.yaml): Docker compose file used to setup and run the Airflow with MLFlow


## Getting Started

Follow the guides in [Instructions.md](Instructions.md) to check the installation process and how to run the pipeline.


## About the recommendation model

The recommendation model can be offline and online.

- `Offline model`: it is an ensemble model over a set of base recommenders with a ranking model on the top. The following base models were considered:
    - `ALS` (Alternating Least Squares) model (uses user-item interactions data)
    - `BPR` (Bayesian Personalized Ranking) model (uses user-item interactions data)
    - `Item2Item` model (uses item-categories data)

    `Item2Item` model is essentially a hand-made model which works with item-features matrix. Given a user ID and user-item interactions data, it computes an average item-features vector and then finds top items according to the similarity criteria. To speed up the computation, the following was used:
    - `TruncatedSVD` algorithm - reduces the number of item features
    - `NearestNeighbors` algorithm - a much faster computation of the closest vectors for a given vector

    The ranking model is automatically selected from these candidates:
    - `CatBoostClassifier`
    - `XGBClassifier`
    - `LGBMClassifier`
    
    These models can easily work with missing data (e.g. some user-item candidates might have only one score from a base model) and can handle different scaling.

    Additionally, user and item features were made - some of them improved the quality of offline recommendations

- `Online model`: it is based on items similarity. `ALS` and `BPR` models were used in order to calculate a set of similar items for each item that was observed when training each model. Hence, the online model can easily recommend similar items based on what items a user have recently interacted with.

Additionally, the most popular items that were at least `added to cart` were retrieved so that they can be recommended for new users or in case it was not possible to give offline recommendations for a user (lack of relevant history data). Also those items were used for building the offline model

## Recommendation model metrics

The resulting recommendation model achived the following metrics:

- `Precision@10`: 0.9%
- `Recall@10`: 5.2%
- `NDCG@10`: 3.7%
- `CoverageItem@10`: 13.6%
- `CoverageUser@10`: 13.6%