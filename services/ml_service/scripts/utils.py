import math
from collections import deque

import yaml
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List

from scripts import logger


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
            content = yaml.safe_load(file)
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
    except Exception as e:
        msg = f"An unexpected error occurred while reading YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_json(path: Path, verbose: bool = True) -> Dict:
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
        logger.error(f"{msg}")
        raise ValueError(msg)

    try:
        with open(path, "r") as file:
            content = json.load(file)
        if verbose:
            logger.info(f"JSON file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"Error parsing JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise json.JSONDecodeError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def save_json(data: Dict, path: Path, verbose: bool = True) -> None:
    """
    Saves a dictionary to a JSON file.

    Args:
        data (Dict):
            The dictionary to be saved.
        path (Path):
            Path to the JSON file where the data will be saved.
        verbose (bool):
            Whether to do any info logs.

    Raises:
        ValueError:
            If the file extension is not .json
        FileNotFoundError:
            If the directory does not exist.
        Exception:
            If there is an error writing the JSON file.
    """
    if path.suffix != ".json":
        msg = f"The file {path} is not a JSON file"
        logger.error(f"{msg}")
        raise ValueError(msg)

    try:
        with open(path, "w") as file:
            json.dump(data, file, indent=4)
        if verbose:
            logger.info(f"JSON file {path} has been saved")
    except FileNotFoundError as e:
        msg = f"Directory {path.parent} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while writing JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def calculate_expire_time(pexpire: int) -> int:
    """
    Calculates the expiration time in seconds from a
    provided expiration time in milliseconds.

    Args:
        pexpire (int):
        The expiration time in milliseconds.

    Returns:
        int:
            The expiration time in seconds, rounded up to the
            nearest integer.
    """
    return math.ceil(pexpire / 1000)


def boolean(arg: Any):
    """
    Converts the given argument to a boolean value.
    Used in the argument parser.

    Args:
        arg (Any):
            The input value to be converted to a boolean.

    Returns:
        bool:
            The boolean value corresponding to the input.

    Raises:
        ValueError:
            If the input value is not "True" or "False".
    """
    arg = str(arg)
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise ValueError("invalid value for boolean argument")


def generate_random_ip() -> str:
    """
    Generates a random IP address as a string.

    Returns:
        str:
            A random IP address in the format "x.x.x.x".
    """
    return (
        f"{random.randint(1, 255)}.{random.randint(0, 255)}."
        f"{random.randint(0, 255)}.{random.randint(0, 255)}"
    )


def get_top_k_items(item_lists: List[List[int]], K: int) -> List[int]:
    """
    Retrieves top K items in circular order given a list of item lists
    preserving the order of the original lists.

    Parameters:
        item_lists (List[List[int]]):
            List of lists with item IDs.
        K (int):
            Number of top items to retrieve.

    Returns:
        list:
            List of top K item IDs.
    """

    queues = [deque(lst) for lst in item_lists]

    recommendations = []
    while len(recommendations) < K and any(queues):
        for q in queues:
            if q and len(recommendations) < K:
                recommendations.append(q.popleft())

    return recommendations


def remove_duplicates(
    lists: List[List[int]], key_index: int = 1, K: int = 10
) -> List[int]:
    """
    Retrieves the top K unique item IDs from a list of sublists,
    preserving the order of the original sublists. Value
    at position `key_index` is used to check for duplicates.
    Only values at position 0 (i.e. itemID) are returned for each
    sublist.

    Parameters:
        lists (List[List[int]]):
            List of sublists.
        key_index (int, optional):
            Index of the key value in each sublist. Defaults to 1.
        K (int, optional):
            Number of top items to retrieve at maximum.
            Defaults to 10.

    Returns:
        List[int]:
            List of top K unique item IDs.
    """
    seen = {}
    result = []
    i = 0
    for item in lists:
        key = item[key_index]
        if key not in seen:
            seen[key] = True
            result.append(item[0])
            i += 1
            if i >= K:
                break
    return result


def str2bool(v: str) -> bool:
    """
    Converts a string value to a boolean value.

    Parameters:
        v (str):
            The string value to be converted to a boolean.

    Returns:
        bool:
            The boolean value corresponding to the input string.

    Raises:
        argparse.ArgumentTypeError:
            If the input string does not represent a valid boolean
            value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
