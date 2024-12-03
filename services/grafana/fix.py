import os

import json
import requests
from dotenv import load_dotenv
from typing import Dict, Union, List, Any


def substitution_datasource_uid(
    dashboard: Union[List[Any], Dict[str, Any], Any],
    current_uid: str,
) -> None:
    """
    Recursively substitutes the datasource UID in the given dashboard object.

    Args:
        dashboard (Union[List[Any], Dict[str, Any], Any]):
            The dashboard object to update.
        current_uid (str):
            The new UID to set for the Prometheus datasource.
    """

    if isinstance(dashboard, dict):
        for k, v in dashboard.items():
            if k == "datasource":
                if v["type"] == "prometheus":
                    v["uid"] = current_uid
            else:
                substitution_datasource_uid(v, current_uid)
    elif isinstance(dashboard, list):
        for elem in dashboard:
            substitution_datasource_uid(elem, current_uid)
    else:
        return


load_dotenv("./services/.env")
USERNAME = os.getenv("GRAFANA_USER")
PASSWORD = os.getenv("GRAFANA_PASS")
HOST = os.getenv("HOST")
PORT = os.getenv("GRAFANA_VM_PORT")
DASHBOARD_FILENAME = "dashboard.json"
FIXED_DASHBOARD_FILENAME = "dashboard_fixed.json"


# URL for Grafana API
url = f"http://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/api/datasources/1"

# Get prometheus datasource data
auth_response = requests.get(url)
prom_datasource = auth_response.json()

# Get datasource uid
current_uid = prom_datasource.get("uid", "")
if not current_uid:
    print(
        "Warning: 'uid' not found in the datasource response. "
        "Using empty string."
    )
print(f"current uid: {current_uid}")

# Read the dashboard
with open(DASHBOARD_FILENAME, "r") as fd:
    dashboard = json.load(fd)

# Change the uid
substitution_datasource_uid(dashboard, current_uid)
print("Fix uid done")

# Save fixed dashboard
with open("dashboard_fixed.json", "w") as fd:
    json.dump(dashboard, fd, indent=2)
