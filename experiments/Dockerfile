FROM apache/airflow:2.7.3-python3.10

USER root

RUN apt-get update
RUN apt-get install -y build-essential

USER airflow

WORKDIR /opt/airflow

COPY dags /opt/airflow/dags
COPY scripts /opt/airflow/scripts
COPY setup.py /opt/airflow/
COPY requirements.txt /opt/airflow/

RUN pip install -U pip && \
    pip install -r requirements.txt && \
    pip install -e .
