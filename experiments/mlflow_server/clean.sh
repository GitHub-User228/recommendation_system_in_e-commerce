#!/bin/bash

mlflow gc --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME