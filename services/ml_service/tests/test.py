"""
This script can be used to use test the fastapi app by sending different
requests from multiple IP addresses or from a single IP address. 

Follow instructions in INSTRUCTIONS.md to test the app properly.

Args:
    --s (int): 
        Stage number. Must be 1 or 2:
            1 - generate test data
            2 - test the app

    --d (bool):
        Whether to use send requests to a service running on Docker
        or Docker-compose.
"""

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import logging
import argparse
import requests
import numpy as np
import pandas as pd
from typing import List

from scripts import logger
from scripts.env import env_vars
from scripts.settings import get_tester_config, config
from scripts.store import SimilarItemsStore, OfflineRecsStore
from scripts.utils import generate_random_ip, save_json, str2bool


class Tester:
    """
    The `Tester` class is responsible for generating and testing the
    FastAPI app by sending different requests from multiple IP
    addresses or a single IP address

    Attributes:
        config (TesterConfig):
            The configuration object
        logger (logging.Logger):
            The logger
    """

    def __init__(self) -> None:
        """
        Initializes the Tester with a given configuration
        """
        self.logger = logging.getLogger(
            f"recsys_service_tester.{self.__class__.__name__}"
        )
        self.config = get_tester_config()

    def generate_users_groups(self) -> List[int]:
        """
        Generates different user groups for testing the FastAPI app
        and saves generated data to a CSV file.

        Returns:
            List[int]:
                A list of user IDs for users with required events data.
        """

        self.logger.info("Generating users groups...")

        n_requests = self.config.n_requests

        # Create an empty DataFrame to store users
        users = pd.DataFrame(columns=["user_id", "group"])

        # Calculate the rate of old users
        rate_offline = (
            self.config.groups_rate.old
            + self.config.groups_rate.old_with_events
        )
        # If it is > 0, then add old users
        if rate_offline > 0:

            # Get old users IDs from OfflineRecsStore and shuffle them
            old_users = (
                pd.DataFrame(
                    list(OfflineRecsStore().recs.keys()), columns=["user_id"]
                )
                .sample(frac=1, random_state=self.config.random_state)
                .reset_index(drop=True)
            )

            # Adjust the number of requests
            n_requests = int(
                min(n_requests * rate_offline, len(old_users)) / rate_offline
            )

            # Keep only the necessary number of users
            old_users = old_users.iloc[: int(rate_offline * n_requests)]

            # Specify `old_with_events` group
            old_users["group"] = "old_with_events"

            # Specify `old` group
            old_users.loc[
                : int(self.config.groups_rate.old * n_requests) - 1, "group"
            ] = "old"

            # Append old users to the main DataFrame
            users = pd.concat([users, old_users], ignore_index=True)
            self.logger.info("Generated `old` groups")

        # Calculate the rate of new users
        rate_new = (
            self.config.groups_rate.new
            + self.config.groups_rate.new_with_events
        )
        # If it is > 0, then add new users
        if rate_new > 0:

            # Generate negative IDs for new users, so that
            # they won't be confused with the old users
            new_users = pd.DataFrame(
                -np.arange(1, int(n_requests * rate_new) + 1),
                columns=["user_id"],
            )

            # Specify `new_with_events` group
            new_users["group"] = "new_with_events"

            # Specify `new` group
            new_users.loc[
                : int(n_requests * self.config.groups_rate.new) - 1, "group"
            ] = "new"

            # Append new users to the main DataFrame
            users = pd.concat([users, new_users], ignore_index=True)
            self.logger.info("Generated `new` groups")

        users["k"] = config["n_recs"]["max"]

        # If the invalid users rate > 0, add them to the DataFrame
        if self.config.groups_rate.invalid > 0:

            # Generate a DataFrame with an invalid `k`
            invalid_users = pd.DataFrame(
                [0] * int(n_requests * self.config.groups_rate.invalid),
                columns=["user_id"],
            )
            invalid_users["group"] = "invalid"
            invalid_users["k"] = config["n_recs"]["max"] * 10

            # Append invalid users to the main DataFrame
            users = pd.concat([users, invalid_users], ignore_index=True)
            self.logger.info("Generated `invalid` group")

        # Shuffle the DataFrame with users if needed
        if self.config.shuffle_requests:
            users = users.sample(frac=1, random_state=self.config.random_state)
            self.logger.info("Shuffled users groups")

        # Generate IPs
        if self.config.multiple_ips:
            users["ip"] = [generate_random_ip() for _ in range(len(users))]
        else:
            users["ip"] = generate_random_ip()

        # Save users to a CSV file
        users.to_csv(Path(Path(__file__).parent, "users.csv"), index=False)
        self.logger.info("Saved users groups")

        # Return user IDs for users with events data
        return users.query("group in ['old_with_events', 'new_with_events']")[
            "user_id"
        ].tolist()

    def generate_events(self, user_ids: List[int]) -> None:
        """
        Generates a dictionary of events for the given user IDs.
        If there are no user IDs provided, the function skips the
        events generation and logs a message. Saves the events to a
        JSON file.

        Args:
            user_ids (List[int]):
                A list of user IDs for which to generate events.
        """

        if len(user_ids) == 0:
            events = {}
            self.logger.info(
                "Skipping events generation since there are no users with "
                "requried events data"
            )
        else:
            self.logger.info("Generating events...")

            items = list(SimilarItemsStore().recs.keys())
            events = {}
            for item_type in config["events_store"]["item_types"]:
                events[item_type] = dict(
                    zip(
                        user_ids,
                        np.random.choice(
                            items,
                            size=len(user_ids)
                            * config["events_store"]["max_events_per_user"],
                            replace=True,
                        )
                        .reshape(
                            (
                                len(user_ids),
                                config["events_store"]["max_events_per_user"],
                            )
                        )
                        .tolist(),
                    ),
                )
            self.logger.info("Generated events")

        save_json(
            data=events,
            path=Path(
                env_vars.events_store_dir, self.config.test_events_filename
            ),
            verbose=False,
        )

    def generate(self) -> None:
        """
        Generates users groups and events data for testing purposes.
        """
        user_ids = self.generate_users_groups()
        self.generate_events(user_ids)

    def test(self, port: int) -> None:
        """
        Tests the recommendation service by sending requests for a set
        of users and logging the responses. Users which were generated
        via `generate_users_groups` function are used.

        Args:
            port (int):
                The port number on which the recommendation service is
                running.
        """

        url = f"http://{env_vars.host}:{port}" f"{config['endpoints']['recs']}"

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "X-Forwarded-For": generate_random_ip(),
        }

        users = pd.read_csv(Path(Path(__file__).parent, "users.csv"))
        self.logger.info("Loaded users groups")

        self.logger.info("Testing...")
        for index, row in users.iterrows():
            headers["X-Forwarded-For"] = row["ip"]
            response = requests.post(
                f"{url}?user_id={row['user_id']}&k={row['k']}",
                headers=headers,
            )
            self.logger.info(
                f"Group: '{row['group']}' | User: {row['user_id']} | IP: '{row['ip']}' | "
                f"Status code: {response.status_code} | Response: {response.json()} "
            )
            time.sleep(self.config.delay)

        self.logger.info("Testing completed")


def main():

    # Adding argument to define the stage and other arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stage", type=int, required=True)
    parser.add_argument("-d", "--docker", type=str2bool)

    # Parsing arguments
    args = parser.parse_args()

    # Executing a specific stage of the Tester
    if args.stage == 1:
        Tester().generate()
    elif args.stage == 2:
        if args.docker == None:
            parser.error("--docker is required when --stage is 2.")
        port = env_vars.app_docker_port
        if args.docker:
            port = env_vars.app_vm_port
        Tester().test(port=port)
    else:
        msg = f"Stage must be 1 or 2, but got {args.stage}"
        logger.error(msg)
        raise ValueError(msg)


if __name__ == "__main__":

    main()
