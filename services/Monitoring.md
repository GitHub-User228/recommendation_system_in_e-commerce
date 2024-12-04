# Monitoring

This file describes the monitoring system used to monitor the ML microservice.

## Description

Grafana's dashboard is used to monitor the ML microservice. A [dashboard.json](/services/grafana/dashboard.json) consists of 15 different panels:
- `CPU Usage`: The CPU usage over time window.
- `RAM Usage`: The memory usage over time window.
- `Successful Requests Increase`: Successful requests count increase over time window for all endpoints.
- `Events Types Total Count`: Number of event types (`view`, `add_to_cart`, `transaction`).
- `Add To Cart Events Ratio`: The ratio of add to cart events. This is essentially the `CTR` metric in the task.
- `Invalid Requests Increase`: Invalid requests count increase over time window for all endpoints.
- `Failed Requests Processing Error Increase`: Increase in the number of times the microservice failed to process the request over time window for all endpoints.
- `Too Many Requests Error Increase`: Increase in the number of requests that exceeded the time limit over time window for all endpoints.
- `Unexpected Error Increase`: Increase in the number of unexpected errors in the microservice over time window for all endpoints.
- `Recommendation Request Duration Quantiles (successful)`: Quantiles of the successful recommendation request processing time over the time window
- `Recommendation Request Duration Quantiles (unsuccessful)`: Quantiles of the unsuccessful recommendation request processing time over the time window
- `Events Store Get Request Duration Quantiles (successful)`: Quantiles of the successful events store get request processing time over the time window
- `Events Store Get Request Duration Quantiles (unsuccessful)`: Quantiles of the unsuccessful events store get request processing time over the time window
- `Events Store Put Request Duration Quantiles (successful)`: Quantiles of the successful events store put request processing time over the time window
- `Events Store Put Request Duration Quantiles (unsuccessful)`: Quantiles of the unsuccessful events store put request processing time over the time window

In order to use this dashboard, you need to specify `prometheus` as a datasource in Grafana UI (specify this url: `http://prometheus:9090`). Then the uid must be fixed in the [dashboard.json](/services/grafana/dashboard.json) - run the following commands to do so:

```bash
# activate the conda environment if you are not already in it
conda activate venv__rss2

# cd to grafana directory
cd services/grafana

# run the python script to fix the datasource uid
python3 -m fix
```

As a result, [dashboard_fixed.json](/services/grafana/dashboard_fixed.json) will be generated. Import it into Grafana and you will see the dashboard via the UI.

Here is a [screenshot](/services/grafana/dashboard.jpg) of the dashboard from Grafana:

![dashboard.jpg](/services/grafana/dashboard.jpg)