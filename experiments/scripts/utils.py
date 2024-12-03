import shutil
import datetime
from pathlib import Path
from collections import deque

import yaml
import json
import mlflow
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict
from tqdm.auto import tqdm
from jinja2 import Template
from mlflow.client import MlflowClient
from jinja2.exceptions import TemplateError


from scripts import logger
from scripts.env import env_vars


def read_yaml(path: Path, verbose: bool = True) -> Dict:
    """
    Reads a yaml file, and returns a dict.

    Args:
        path_to_yaml (Path):
            Path to the yaml file

    Returns:
        Dict:
            The yaml content as a dict.
        verbose:
            Whether to do any info logs

    Raises:
        ValueError:
            If the file is not a YAML file
        FileNotFoundError:
            If the file is not found.
        yaml.YAMLError:
            If there is an error parsing the yaml file.
    """
    if path.suffix not in [".yaml", ".yml"]:
        msg = f"The file {path} is not a YAML file"
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)
    try:
        with open(path, "r") as file:
            template = Template(file.read())
        rendered_yaml = template.render(dict(env_vars))
        content = yaml.safe_load(rendered_yaml)
        if verbose:
            logger.info(f"YAML file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except TemplateError as e:
        msg = f"Error rendering YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise TemplateError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_json(
    path: Path, verbose: bool = True, keys_to_integer: bool = False
) -> Dict:
    """
    Reads a JSON file and returns a dict.

    Args:
        path (Path):
            Path to the JSON file
        verbose (bool):
            Whether to do any info logs

    Returns:
        Dict:
            The JSON content as a dict.

    Raises:
        ValueError:
            If the file is not a JSON file
        FileNotFoundError:
            If the file is not found.
        json.JSONDecodeError:
            If there is an error parsing the JSON file.
    """
    if path.suffix != ".json":
        msg = f"The file {path} is not a JSON file"
        logger.error(msg)
        raise ValueError(msg)

    try:
        with open(path, "r") as file:
            content = json.load(file)
        if verbose:
            logger.info(f"JSON file {path} has been loaded")
        if keys_to_integer:
            content = {int(k): v for k, v in content.items()}
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"Error parsing JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise json.JSONDecodeError(msg) from e
    except ValueError:
        msg = "Could not convert non-integer keys to integers in JSON file"
        logger.error(msg)
        raise ValueError(msg)
    except Exception as e:
        msg = f"An unexpected error occurred while reading JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_pkl(path: Path) -> object:
    """
    Reads a model object from a file using joblib.

    Args:
        path (Path):
            The path to the file with the model to load.

    Returns:
        object:
            The loaded model object.

    Raises:
        ValueError:
            If the file does not have a .pkl extension.
        FileNotFoundError:
            If the file does not exist.
        IOError:
            If an I/O error occurs during the loading process.
        Exception:
            If an unexpected error occurs while loading the model.
    """

    if path.suffix != ".pkl":
        msg = f"The file {path} is not a pkl file"
        logger.error(msg)
        raise ValueError(msg)

    try:
        with open(path, "rb") as f:
            model = joblib.load(f)
        logger.info(f"Model {path} has been loaded")
        return model
    except FileNotFoundError as e:
        msg = f"File '{path}' does not exist"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except IOError as e:
        msg = f"An I/O error occurred while loading a model from {path}"
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while loading a model from {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def save_yaml(path: Path, data: Dict, verbose: bool = True) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        path (Path):
            Path to the YAML file where the data will be written.
        data (Dict):
            The dictionary content to write to the YAML file.
        verbose (bool, optional):
            Whether to log informational messages. Defaults to True.

    Raises:
        ValueError:
            If the file is not a YAML file.
        IOError:
            If there is an error writing to the file.
        yaml.YAMLError:
            If there is an error serializing the dictionary to YAML.
        Exception:
            If an unexpected error occurs.
    """
    if path.suffix not in [".yaml", ".yml"]:
        msg = f"The file {path} is not a YAML file."
        logger.error(msg)
        raise ValueError(msg)

    try:
        with open(path, "w") as file:
            yaml.safe_dump(data, file, sort_keys=False)
        if verbose:
            logger.info(f"YAML file {path} has been written successfully.")
    except IOError as e:
        msg = f"Error writing to file {path}."
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error serializing data to YAML file {path}."
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while writing YAML file {path}."
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def save_json(path: Path, data: Dict, verbose: bool = True) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        path (Path):
            Path to the JSON file to write.
        data (Dict):
            The dictionary data to write to the file.
        verbose (bool):
            Whether to log informational messages.

    Returns:
        None

    Raises:
        ValueError:
            If the file extension is not '.json'.
        FileNotFoundError:
            If the directory does not exist.
        Exception:
            If an unexpected error occurs during the write operation.
    """
    if path.suffix != ".json":
        msg = f"The file {path} does not have a .json extension"
        logger.error(msg)
        raise ValueError(msg)

    try:
        # Ensure the directory exists
        if not path.parent.exists():
            msg = f"Directory {path.parent} does not exist"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        if verbose:
            logger.info(f"JSON file {path} has been saved")
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while writing JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def save_pkl(path: Path, model: object) -> None:
    """
    Saves a model object to a file using joblib.

    Args:
        path (Path):
            The path to the file where the model will be saved.
        model (object):
            The model object to save.

    Raises:
        ValueError:
            If the file does not have a .pkl extension.
        IOError:
            If an I/O error occurs during the saving process.
        Exception:
            If an unexpected error occurs while saving the model.
    """
    if path.suffix != ".pkl":
        msg = f"The file {path} is not a pkl file"
        logger.error(msg)
        raise ValueError(msg)

    try:
        with open(path, "wb") as f:
            joblib.dump(model, f)
        logger.info(f"Model has been saved to {path}")
    except IOError as e:
        msg = f"An I/O error occurred while saving a model to {path}"
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while saving a model to {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def reduce_size(df: pd.DataFrame) -> None:
    """
    Reduces the size of the DataFrame by converting integer
    and float columns to smaller data types.

    This function iterates through each column in the DataFrame and
    checks the minimum and maximum values. It then converts the column
    to a smaller data type if possible, such as `uint8`, `uint16`,
    `int8`, or `int16`, to reduce the memory footprint of the DataFrame.

    Args:
        df (pd.DataFrame):
            The DataFrame to be reduced in size.
    """
    print(
        "Dataframe memory usage before optimisation:",
        df.memory_usage().sum() / (1024**2),
        "MB",
    )
    for col in tqdm(df.columns):
        if "int" in df[col].dtype.name:
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if max(abs(df[col].min()), df[col].max()) <= 127:
                    df[col] = df[col].astype("int8")
                elif max(abs(df[col].min()), df[col].max()) <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")
        elif "float" in df[col].dtype.name:
            df[col] = df[col].astype("float32")
    print(
        "Dataframe memory usage after optimisation:",
        df.memory_usage().sum() / (1024**2),
        "MB",
    )


def get_bins(x: int) -> int:
    """
    Calculates the appropriate number of bins for the histogram
    according to the number of the observations

    Args:
        x (int):
            Number of the observations

    Returns:
        int:
            Number of bins
    """
    if x > 0:
        n_bins = max(int(1 + 3.2 * np.log(x)), int(1.72 * x ** (1 / 3)))
    else:
        msg = (
            "An invalid input value passed. Expected a positive "
            + "integer, but got {x}"
        )
        logger.error(f"{msg}")
        raise ValueError(msg)
    return n_bins


def get_experiment_id(
    experiment_name: str, client: mlflow.tracking.client.MlflowClient
) -> str:
    """
    Get the ID of an MLflow experiment, creating a new one if it
    doesn't exist. Also restores the experiment if it is not active.

    Args:
        experiment_name (str):
            The name of the MLflow experiment.
        client (mlflow.tracking.client.MlflowClient):
            The MLflow client to use for interacting with the MLflow
            tracking server.

    Returns:
        str:
            The ID of the MLflow experiment.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.info(
            f"Experiment '{experiment_name}' not found. "
            "Creating a new experiment..."
        )
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception as e:
            msg = f"Failed to create experiment '{experiment_name}'"
            logger.error(msg)
            raise Exception(msg) from e
    else:
        logger.info(f"Experiment '{experiment_name}' exists")
        experiment_id = experiment.experiment_id

    if experiment.lifecycle_stage != "active":
        logger.info(
            f"Experiment '{experiment_name}' is not active. Current state: "
            f"{experiment.lifecycle_stage}. Restoring..."
        )
        client.restore_experiment(experiment_id)

    return experiment_id


def spark_parquet_to_single_parquet(path: Path, filename: str) -> None:
    """
    Moves a single Parquet files in a directory into a single
    Parquet file in the parent directory

    Args:
        path (Path):
            The path to the directory containing the Parquet file to
            move.
        filename (str):
            The name of the Parquet file to move.
    """
    path1 = Path(path, filename)
    files = list(path1.glob("*.parquet"))
    if len(files) == 0:
        msg = f"No Parquet files found in {path1}"
        logger.error(msg)
        raise ValueError(msg)
    elif len(files) > 1:
        msg = f"Multiple Parquet files found in {path1}"
        logger.error(msg)
        raise ValueError(msg)
    path2 = Path(path, filename, files[0])
    path3 = Path(path, "tmp.parquet")
    shutil.move(path2, path3)
    shutil.rmtree(path1)
    shutil.move(path3, path1)


def get_runs_data(
    experiment_name: str, is_airflow: bool = False
) -> pd.DataFrame:
    """
    Retrieves the runs data for the experiment specified in the components.yaml file.

    Returns:
        pd.DataFrame:
            A DataFrame containing the runs data for the specified experiment.
        is_airflow (bool):
            Whether the experiment is being run in Airflow.
    """

    host = "localhost"
    if is_airflow:
        host = env_vars.mlflow_server_host

    uri = f"http://{host}:{env_vars.mlflow_server_port}"

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()

    # Get the experiment id
    experiment_id = get_experiment_id(
        experiment_name=experiment_name, client=client
    )

    runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

    return runs_df


def calculate_node_levels(graph: nx.Graph) -> Dict[int, int]:
    """
    Assign levels to each node in a DAG. Root nodes are at level 0.

    Args:
        graph (nx.Graph):
            The DAG graph

    Returns:
        Dict[int, int]:
            Dictionary which maps node id to level
    """
    levels = {}
    in_degree = {node: graph.in_degree(node) for node in graph.nodes()}
    queue = deque([node for node, degree in in_degree.items() if degree == 0])

    for root in queue:
        levels[root] = 0

    while queue:
        current = queue.popleft()
        current_level = levels[current]
        for neighbor in graph.successors(current):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)

    return levels


def create_new_experiment(
    experiment_name_prefix: str, is_airflow: bool = False
) -> None:
    """
    Creates a new MLflow experiment with a unique name based on the
    current date and time.

    Args:
        experiment_name_prefix (str):
            The prefix for the new experiment name.
        is_airflow (bool, optional):
            Whether the function is being called from Airflow.
            This is used to determine the host for the MLflow server.
            Defaults to False.
    """

    # Get the host
    host = "localhost"
    if is_airflow:
        host = env_vars.mlflow_server_host

    # Setup the MLFlow client
    mlflow.set_tracking_uri(f"http://{host}:{env_vars.mlflow_server_port}")
    mlflow.set_registry_uri(f"http://{host}:{env_vars.mlflow_server_port}")
    client = MlflowClient()

    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Create new experiment name
    experiment_name = experiment_name_prefix + "__" + current_date

    # Update the experiment name with the current date
    save_yaml(
        data={"experiment_name": experiment_name},
        path=Path(env_vars.config_dir, "experiment_name.yaml"),
    )

    # Create new experiment
    experiment_id = get_experiment_id(
        experiment_name=experiment_name, client=client
    )
