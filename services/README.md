# Music Recommendation System

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/grafana-%23F46800.svg?style=for-the-badge&logo=grafana&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pydantic](https://img.shields.io/badge/Pydantic-CC0066?style=for-the-badge)
![Uvicorn](https://img.shields.io/badge/Uvicorn-223366?style=for-the-badge)


## Description

This part of the repository contains info on how to create and run a ML microservice for a music recommendation system. It is implemented using FastAPI and Python. The microservice can be deployed in the following configurations:
- `conda environment`
- `docker container`
- `docker compose`

In case of `docker compose` deployment, the following services are also available:
- `Redis` - used for limiting the number of requests
- `Prometheus` - used for monitoring the service by collecting metrics
- `Grafana` - used for monitoring the service by visualizing the metrics

## Project Structure

[Instructions.md](Instructions.md): Info on how to run the microservice

[Monitoring.md](Monitoring.md): Info on how to monitor the microservice

**[prometheus](/services/prometheus/)**: This directory contains files related to monitoring the service by using Prometheus
- [prometheus.yml](/services/prometheus/prometheus.yml): Configuration file for Prometheus

**[grafana](/services/grafana)**: This directory contains files related to monitoring the service by using Grafana
- [dashboard.json](/services/grafana/dashboard.json): Dashboard for Grafana
- [dashboard_fixed.json](/services/grafana/dashboard.json): Fixed dashboard for Grafana
- [dashboard.jpg](/services/grafana/dashboard.jpg): Screenshot of the Grafana dashboard
- [fix.py](/services/grafana/fix.py): Script for fixing the datasource UID in the Grafana dashboard

**[ml_service](/services/ml_service)**: This directory contains files related to the ML microservice:

- **[data](/services/ml_service/data)**: This directory contains data files with recommendations:
    - [offline](/services/ml_service/data/offline): Directory with offline recommendations data
    - [online](/services/ml_service/data/online): Directory with online recommendations data
    - [popular](/services/ml_service/data/popular): Directory with popular recommendations data
- **[config](/services/ml_service/config)**: Configuration files for the ML microservice
    - [config.yaml](/services/ml_service/config/config.yaml): Configuration for the project
    - [logger_config.yaml](/services/ml_service/config/logger_config.yaml): Configuration for the logger
- **[entrypoints](/services/ml_service/entrypoints)**: This directory contains files related to the ML microservice entrypoints:
    - [entrypoint_single.sh](/services/ml_service/entrypoints/entrypoint_single.sh): Entrypoint for the ML microservice in the `docker container` configuration
    - [entrypoint_compose.sh](/services/ml_service/entrypoints/entrypoint_compose.sh): Entrypoint for the ML microservice in the `docker compose` configuration
- **[scripts](/services/ml_service/scripts)**: This directory contains Python scripts used by ML microservice
- **[app](/services/ml_service/app)**: This directory contains FastAPI apps as Python files:
    - [app_simple.py](/services/ml_service/app/app_simple.py): FastAPI app for the ML microservice in the `docker container` or `conda environment` configuration
    - [app.py](/services/ml_service/app/app.py): FastAPI app for the ML microservice in the `docker compose` configuration
- **[tests](/services/ml_service/tests)**: This directory contains tests for the ML microservice:
    - [test.py](/services/ml_service/tests/test.py): Tests for the ML microservice in all configurations by sending requests to it
- [requirements.txt](/services/ml_service/requirements.txt): List of required Python packages
- [Dockerfile](/services/ml_service/Dockerfile): Dockerfile used to build the docker image for the ML microservice in the `docker compose` configuration
- [setup.py](/services/ml_service/setup.py): Setup file for the ML microservice, so that all scripts can be easily accessable from any directory

[.env_template](/services/.env_template): This is a template for the .env file with environment variables  

[docker-compose.yaml](/services/docker-compose.yaml): Docker compose file used to run all the services  

[Dockerfile_single_service](/services/Dockerfile_single_service): Dockerfile used to build the docker image for a microservice in the `docker container` configuration

## Getting Started

Follow the guides in [Instructions.md](Instructions.md) to run the microservice.

## How the recommendations are made

The microservice must receive the following parameters in order to make recommendations:
- `user_id` - user ID
- `k` - number of recommendations to be made (must be within the defined range)

There are 4 different combinations of recommendations the user can receive:
1. Offline only recommendations - the user has history data, but have not yet interacted with new items
2. Popular only recommendations - the user has no history data and has not yet interacted with new items
3. Offline + online recommendations - the user has history data and has interacted with new items (has data in events store)
4. Popular + online recommendations - the user has no history data and has interacted with new items (has data in events store)

In case of an invalid request or some errors in the microservice, the corresponding error response will be returned.

Since there are multiple sources of recommendations, it is essential to specify the way to combine them:
1. top `k` items according to similarity score are retrieved from online recommendations of a single source given a list of items from events store
2. top `k` items are retrieved in the round-robin fashion (preserving the original order) from different sources of online recommendations after applying the first step for each one
3. top `k` items are retrieved by combining the result of step 2 (online recs) and either offline or popular recommendations in the round-robin fashion. Online recs are given higher priority in any case (e.g. the first item in the recs is from online recommendations)


## TODO

- Enhance the monitoing system by tracking `view`, `add to cart`, `transaction` events separately, possibly using a different events store for each type of event.
- In case of enhanced monitoring, reconsider how the online recommendations are made. For example, mix items from different events store via round-robin fashion prioritizing `transaction` events over `view` and `add to cart` events, `add to cart` events over `view` events.
- In case of enhanced monitoring, reconsider the `put` request structure by adding a new field for the type of event.