import os
from pathlib import Path

import yaml
import logging.config
from jinja2 import Template
from jinja2.exceptions import TemplateError

from scripts.env import env_vars


try:
    with open(Path(f"{env_vars.config_dir}/logger_config.yaml"), "r") as file:
        template = Template(file.read())
        rendered_yaml = template.render(LOG_DIR=env_vars.log_dir)
        config = yaml.safe_load(rendered_yaml)
except FileNotFoundError as e:
    raise FileNotFoundError(
        "Unable to locate the logging configuration file. "
    ) from e
except yaml.YAMLError as e:
    raise yaml.YAMLError(f"Error parsing logging configuration file") from e
except TemplateError as e:
    raise TemplateError(f"Error rendering logging configuration file") from e
except Exception as e:
    raise Exception(
        f"Unexpected error occurred while reading logging configuration file"
    ) from e

# # Check and create log directory if not exists
# os.makedirs(env_vars.log_dir, exist_ok=True)

# Basic logging configuration
logging.config.dictConfig(config)

# Create and provide logger instance
logger = logging.getLogger("recsys_experiments")
