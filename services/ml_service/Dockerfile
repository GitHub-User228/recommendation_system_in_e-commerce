FROM python:3.10.15-slim

LABEL author=${LABEL}

WORKDIR /fastapi_app/ml_service

COPY . .

RUN pip install -r requirements.txt && \
    pip install -e .

ENTRYPOINT ["/bin/bash", "entrypoints/entrypoint_compose.sh"]