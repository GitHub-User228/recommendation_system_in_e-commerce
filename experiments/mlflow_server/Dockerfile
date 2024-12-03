FROM python:3.10.15-slim

COPY requirements.txt /mlflow_server/requirements.txt

RUN pip install -r /mlflow_server/requirements.txt

ENTRYPOINT [ "bash", "/mlflow_server/start.sh" ]