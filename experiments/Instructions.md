# Instructions on how to run the pipeline

Each instruction is executed from the repository directory

## 1. Installation

Fill in the `.env_template` file with the necessary environment variables and rename it to `.env`.

Install `Conda` on the machine and initialize it.

Now, follow the instructions below to create the conda environment and install the dependencies.

```python

# creating a conda environment
conda create -y --name venv__rss python=3.10.15

# activating the conda environment
conda activate venv__rss

# exporting the environment variables for the .env
conda env config vars set $(cat experiments/.env | tr '\n' ' ')

# reactivating the conda environment
conda deactivate
conda activate venv__rss

# cd to the directory with the experiments
cd experiments

# installing the dependencies
pip install -r requirements.txt
pip install -e .
```

## 2. Setup and Start the Airflow with MLFlow

Follow the instructions below to setup and start the Airflow with MLFlow:

```bash

# cd to the directory with the experiments
cd experiments

# Save AIRFLOW_UID of your machine to the .env file
echo -e "\nAIRFLOW_UID=$(id -u)" >> .env 

# Setup the Airflow
docker compose up airflow-init 

# Clear possible cache that resulted from the last step
docker compose down --volumes --remove-orphans

# Start the Airflow with MLFlow
docker compose up --build

```

## 3. Run the pipeline

Log in into the Airflow UI using the following credentials:
- username: `airflow`
- password: `airflow`

Run the DAG by the name `recsys_master`.

The first stage of the pipeline is used in order to evaluate the recsys system on the current data. The second stage is used to generate offline recommendations, similar items data and retrieve popular items data using all available data and fitted ensemble model from the first pipeline.

Also see [components.yaml](/experiments/config/components.yaml) and [settings.py](/experiments/scripts/settings.py) where you can modify the configuration of the components.

Check [airflow.yaml](/experiments/config/airflow.yaml) for the configuration of the DAG.

Here is the screenshot of the DAG from the Airflow UI:
![DAG](/experiments/assets/DAG.jpg)