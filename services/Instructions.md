# Instructions on how to start and use the microservice

Each instruction is executed from the repository directory.

To begin with, adjust variables in the `.env_template` file (if you need) and rename it to `.env`.

Then perform the following imports of the data:
- `recommendations.parquet` with final offline recs to [offline](/services/ml_service/data/offline) directory.
- `similar_items_als.parquet` with similar items data to [online](/services/ml_service/data/online) directory.
- `similar_items_bpr.parquet` with similar items data to [online](/services/ml_service/data/online) directory.
- `top_items.parquet` with popular items data to [popular](/services/ml_service/data/popular) directory.

For convinience, you can download all parquet files from [here](https://drive.google.com/drive/folders/175MxkKviccgTmI1UljsntAVfn0OYZNlc?usp=sharing)

## 1. FastAPI microservice via conda environment

Install Conda on the machine and initialize it before proceeding.

```bash
# create a conda environment
conda create -y --name venv__rss2 python=3.10.15

# activate the conda environment
conda activate venv__rss2

# add UID and GID to the .env file
echo "UID=$(id -u)" >> services/.env
echo "GID=$(id -g)" >> services/.env

# export the environment variables from the .env
conda env config vars set $(cat services/.env | tr '\n' ' ')

# reactive the conda environment
conda deactivate
conda activate venv__rss2

# cd to the directory with the ml service
cd services/ml_service

# install the dependencies
pip install -r requirements.txt
pip install -e .

# start the microservice via uvicorn
uvicorn app.app_simple:app --reload --host ${HOST} --port ${APP_DOCKER_PORT}
```

### Example curl query to the microservice

```bash
curl -X 'POST' 'http://localhost:8081/recommend?user_id=507255&k=5' -H 'accept: application/json'
```

This query should be valid and return the following response: 
```bash
{
  "user_id": 507255,
  "item_ids": [
    313107,
    11478,
    390485,
    74945,
    78609
  ]
}
```

Make sure to stop the running microservice, so that the `APP_DOCKER_PORT` port is available for the next step.

```bash
# stop the service (press this in the terminal where the service is running)
Ctrl + C
```

Do not delete this conda environment, since it will be used in the future step.


## 2. FastAPI microservice via Docker container

```bash
# cd to the directory with the services
cd services

# build an image from Dockerfile
docker build -t ml_service_image -f Dockerfile_single_service .

# run the container from the created image
docker run --name ml_service_container --publish ${APP_VM_PORT}:${APP_DOCKER_PORT} --volume=./ml_service/data:/fastapi_app/ml_service/data --env-file .env ml_service_image
```

### Example curl query to the microservice

```bash
curl -X 'POST' 'http://localhost:4602/recommend?user_id=507255&k=5' -H 'accept: application/json'
```

This query should be valid and return the following response: 
```bash
{
  "user_id": 507255,
  "item_ids": [
    313107,
    11478,
    390485,
    74945,
    78609
  ]
}
```

### Stopping the docker container

It is necessary to stop the container before procceding to the next step since the ports used in this case are required for the next step and need to be free.

```bash
# Run this in another terminal
docker container stop ml_service_container
```

## 3. Docker compose for microservice and monitoring system

```bash
# cd to the directory with the services
cd services

# build the image and starting the services via docker compose
docker compose up --build
```

### Example curl query to the microservice

```bash
curl -X 'POST' 'http://localhost:4602/recommend?user_id=507255&k=5' -H 'accept: application/json'
```

This query should be valid and return the following response unless you exceeded the maximum number of requests from a single IP or the global limit:
```bash
{
  "user_id": 507255,
  "item_ids": [
    313107,
    11478,
    390485,
    74945,
    78609
  ]
}
```

In case you exceeded the limit, you will see the response similar to one of the following: 
- `{"detail":"Too Many Requests from your IP. Retry after 14 seconds."}`
- `{"detail":"Too Many Overall Requests. Retry after 14 seconds."}`


## 4. Simulation of the load on the microservice

Script [test.py](/services/ml_service/tests/test.py) simulates a load on the microservice. You can adjust different parameters in the [config.yaml](/services/ml_service/config/config.yaml) file in the `tester` section:
- `n_requests` - number of requests to send
- `delay` - delay between requests in seconds
- `use_different_ip` - whether to use a different IP address for each request
- `shuffle_requests` - whether to shuffle the requests order
- `random_state` - random state for reproducibility
- `groups_rate` - the rate of different kinds of requests

In order to test the microservice, change `events_store.is_testing` to `True` in the [config.yaml](/services/ml_service/config/config.yaml) file. Then follow the instructions below:

```bash
# If you are not yet in the conda environment created eariler, activate it
conda activate venv__rss2

# Generate test data
python3 -m tests.test --stage 1

# Restart the microservice (can be of any type)

# Test the microservice (change --docker to False if testing conda-based microservice)
python3 -m tests.test --stage 2 --docker True
```

You should be able to see the logs in the [test_service.log](/services/ml_service/logs/test_service.log).

It is important to note that the script tests only the load on the `/recommend` and `/get` endpoints, but not the `/put` endpoint, which is used for adding new events (items for a user) to the events store. In order to test the `/put` endpoint, you can run the following command:

```bash
curl -X 'POST' 'http://localhost:4602/put?user_id=1&item_id=12&item_type=add_to_cart' -H 'accept: application/json'
```
You should recieve `{"result":"ok"}` in case of successful request.

---

Addresses of the services:
- microservice: [http://localhost:4602](http://localhost:4602)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000)