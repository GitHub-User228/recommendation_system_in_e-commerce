import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scripts import logger
from scripts.utils import calculate_node_levels
from scripts.components.base import BaseComponent
from scripts.plotters import custom_hist_multiplot
from scripts.settings import get_eda_component_config

sns.set_style("dark")
sns.set_theme(style="darkgrid", palette="deep")


class EDAComponent(BaseComponent):
    """
    Performs exploratory data analysis (EDA) on the training data, including:
    - user analysis
    - item analysis
    - item features analysis

    Attributes:
        config (EDAComponentConfig):
            Class containing the configuration parameters.
        show_graphs (bool):
            Whether to display graphs during the analysis.
        is_testing (bool):
            Whether the component is being used for testing.
        is_airflow (bool):
            Whether the component is being used via Airflow.
            This is used to determine the host for the MLflow server.
    """

    def __init__(
        self, show_graphs: bool = True, is_airflow: bool = False
    ) -> None:
        """
        Initializes the EDAComponent class with the necessary
        configuration.

        Args:
            show_graphs (bool):
                Whether to display graphs during the analysis.
            is_airflow (bool, optional):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
                Defaults to False.
        """
        self.config = get_eda_component_config()
        self.show_graphs = show_graphs
        self.is_testing = False
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def read_parquet(
        self, filename: str, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Reads a Parquet file from the configured source path and
        returns a DataFrame.

        Args:
            filename (str):
                The name of the Parquet file to read.
            verbose (bool):
                Whether to log the progress of reading the file.

        Returns:
            pd.DataFrame:
                The Spark DataFrame containing the data.
        """
        df = pd.read_parquet(Path(self.config.source_path, filename))
        if verbose:
            logger.info(f"Read {filename} from {self.config.source_path}.")
        return df

    def default_analysis(
        self,
        events: pd.DataFrame,
        user: bool = True,
    ) -> None:
        """
        Performs a default analysis on a given DataFrame of events, including:
        - Number of items/users per user/item
        - Cumulative number of items/users over time
        - Number of items/users per month

        Args:
            events (pd.DataFrame):
                The DataFrame of events to analyze.
            user (bool):
                Whether to analyze users or items.
                Defaults to True.
        """

        feat1 = "user" if user else "item"
        feat2 = "item" if user else "user"

        for rating in events[self.config.rating_col].unique():

            ### Number of tracks per user (users per track) for specific rating
            print(
                f"\n----- {feat2.upper()}S PER {feat1.upper()} FOR RATING {rating} -----\n"
            )
            col = f"{feat2}s_per_{feat1}"
            df = (
                events.query(f"{self.config.rating_col} == {rating}")
                .groupby(self.config.fields_id[feat1])
                .agg({self.config.fields_id[feat2]: "count"})
                .rename(columns={self.config.fields_id[feat2]: col})
                .sort_values(by=col, ascending=False)
            )
            df.head(10).to_csv(
                Path(
                    self.config.data_path,
                    f"{feat1}s_per_{feat2}_top_rating_{rating}.csv",
                )
            )
            print(f"Number of unique {feat1}s: {df.shape[0]}\n")
            print(f"Quantiles")
            q = (
                df[col]
                .quantile(self.config.quantiles)
                .reset_index()
                .rename(columns={"index": "quantiles"})
            )
            print(q.to_markdown(), end="\n")
            q.to_csv(
                Path(
                    self.config.data_path,
                    f"{feat1}s_per_{feat2}_quantiles_rating_{rating}.csv",
                ),
                index=False,
            )
            custom_hist_multiplot(
                data=df,
                columns=[col],
                stat="count",
                title=f"Number of {feat2}s per {feat1} and rating {rating}",
                width_factor=0.8,
                height_factor=0.8,
                title_fontsize=16,
                kde=False,
                yscale="log",
                savepath=Path(
                    self.config.assets_path,
                    f"{feat2}s_per_{feat1}_rating_{rating}_hist.png",
                ),
                show=self.show_graphs,
            )

            ### Cumulative number of users (items) over time
            col = f"number_of_{feat1}s"
            df = (
                events.query(f"{self.config.rating_col} == {rating}")
                .groupby(self.config.fields_id[feat1])
                .agg({"date": "min"})
                .reset_index()
                .sort_values("date")
                .groupby("date")
                .size()
                .reset_index(name=col)
            )
            df[col] = df[col].cumsum()

            fig = plt.figure(figsize=(15, 3))
            sns.lineplot(data=df, x="date", y=col, lw=5)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.title(
                f"Cumulative number of {feat1}s over time for rating {rating}",
                fontsize=16,
            )
            if self.show_graphs:
                plt.show()
            fig.savefig(
                Path(
                    self.config.assets_path,
                    f"cumulative_{feat1}_count_rating_{rating}.png",
                ),
                bbox_inches="tight",
            )

            ### Number of users (items) per day
            df = (
                events.query(f"{self.config.rating_col} == {rating}")
                .groupby("date")
                .agg({self.config.fields_id[feat1]: pd.Series.nunique})
                .reset_index()
                .rename(columns={self.config.fields_id[feat1]: col})
            )

            fig = plt.figure(figsize=(15, 3))
            sns.lineplot(data=df, x="date", y=col, lw=3)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.title(
                f"Number of {feat1}s per day for rating {rating}", fontsize=16
            )
            if self.show_graphs:
                plt.show()
            fig.savefig(
                Path(
                    self.config.assets_path,
                    f"{feat1}_count_per_day_rating_{rating}.png",
                ),
                bbox_inches="tight",
            )

    def item_analysis(self) -> None:
        """
        Performs an item-level analysis on the training events data.
        """
        events = self.read_parquet(
            filename=self.config.events_filenames["train"]
        )
        events["date"] = pd.to_datetime(
            events[self.config.date_col].dt.floor("D").astype("datetime64[s]")
        )
        self.default_analysis(events, user=False)
        logger.info(f"Finished item analysis")

    def user_analysis(self) -> None:
        """
        Performs a user-level analysis on the training events data.
        """
        events = self.read_parquet(
            filename=self.config.events_filenames["train"]
        )
        events["date"] = pd.to_datetime(
            events[self.config.date_col].dt.floor("D").astype("datetime64[s]")
        )
        self.default_analysis(events, user=True)
        logger.info(f"Finished user analysis")

    def category_tree_analysis(self) -> Tuple[List[int], nx.DiGraph]:
        """
        Performs an analysis of the item category tree.

        Returns:
            List[int]:
                A list of nodes depth with respect to the roots.
            nx.DiGraph:
                A directed graph representing the item category tree.
        """

        # Read data
        df = self.read_parquet(filename=self.config.category_tree_filename)

        # Separate isolated nodes
        isolated_nodes = list(
            df[df[self.config.fields_id["parent_item_category"]].isnull()][
                self.config.fields_id["item_category"]
            ].values
        )
        df = df.dropna().astype("uint32")

        # Create a graph from edges
        G = nx.from_pandas_edgelist(
            df=df,
            source=self.config.fields_id["parent_item_category"],
            target=self.config.fields_id["item_category"],
            create_using=nx.DiGraph(),
        )

        # Add isolated nodes
        G.add_nodes_from(isolated_nodes)

        # Print the number of unique categories
        print(f"Number of unique categories: {len(G.nodes())}")

        # Check for cycles
        try:
            cycles = list(nx.find_cycle(G, orientation="original"))
            print("Cycles detected in the item category tree:", cycles)
        except nx.exception.NetworkXNoCycle:
            print("No cycles detected in the item category tree.")

        # Identify weakly connected components
        wcc = list(nx.weakly_connected_components(G))

        # Number of nodes in the components
        fig = plt.figure(figsize=(12, 5))
        sns.barplot(
            sorted(map(len, wcc)),
            orient="h",
            alpha=0.65,
        )
        plt.xlabel("Number of categories in a component")
        plt.grid(axis="y", linestyle="")
        if self.show_graphs:
            plt.show()
        fig.savefig(
            Path(
                self.config.assets_path,
                f"category_tree_component_size_distribution.png",
            ),
            bbox_inches="tight",
        )

        # Calculate the level (depth) of each node
        if nx.is_directed_acyclic_graph(G):

            node_levels = calculate_node_levels(G)

            custom_hist_multiplot(
                data=pd.DataFrame(
                    node_levels.items(),
                    columns=[
                        self.config.fields_id["item_category"],
                        "Number of categories per level",
                    ],
                ),
                columns=["Number of categories per level"],
                kde=False,
                features_kind="cat",
                cat_orient="h",
                stat="count",
                height_factor=0.7,
                savepath=Path(
                    self.config.assets_path,
                    f"category_tree_node_level_distribution.png",
                ),
                show=self.show_graphs,
            )

            # Print root categories
            root_categories = [
                node for node in G.nodes() if G.in_degree(node) == 0
            ]
            print(f"Number of root categories: {len(root_categories)}")
            print(f"Root categories: {root_categories}")

            # Print leaf categories
            leaf_categories = [
                node for node in G.nodes() if G.out_degree(node) == 0
            ]
            print(f"Number of leaf categories: {len(leaf_categories)}")

        return node_levels, G

    def item_availability_analysis(self) -> None:
        """
        Performs an analysis of the item availability.
        """

        # Read data
        df = self.read_parquet(
            filename=self.config.item_features_filenames["availability"]
        )

        # Number of days item was available or not available
        n_days = (
            df.groupby([self.config.fields_id["item"], "available"])["days"]
            .sum()
            .rename("number of days")
            .reset_index()
        )

        for i in ["0", "1"]:

            title = "Number of days an item was"
            if i == "0":
                title += " not available"
            else:
                title += " available"

            # Filter data based on availability status
            df2 = n_days.query(f'available == "{i}"')

            custom_hist_multiplot(
                data=df2,
                columns=["number of days"],
                kde=False,
                features_kind="num",
                stat="count",
                height_factor=0.7,
                savepath=Path(
                    self.config.assets_path,
                    f"days_item_availability_{i}_hist.png",
                ),
                show=self.show_graphs,
                title=title,
            )

    def item_category_analysis(
        self,
        G: nx.DiGraph,
        category_levels: dict[str, int],
    ) -> None:
        """
        Performs an analysis of the item category.

        Args:
            G (nx.DiGraph):
                A directed graph representing the item category tree.
            category_levels (List[int]):
                A list of categories depth with respect to the root
                categories.
        """

        # Read data
        df = self.read_parquet(
            filename=self.config.item_features_filenames["category"]
        )

        # Find the level of the category for each item category
        df["category_level"] = df[self.config.fields_id["item_category"]].map(
            category_levels
        )

        # Node to root
        node_root = {}
        for node in G.nodes():
            ancestors = nx.ancestors(G, node)
            ancestors.add(node)
            root = [node for node in ancestors if G.in_degree(node) == 0][0]
            node_root[node] = root
        df["root_category"] = df[self.config.fields_id["item_category"]].map(
            node_root
        )

        # Get nodes that are not in the category tree
        nodes_not_in_tree = [
            node
            for node in df[self.config.fields_id["item_category"]].unique()
            if node not in G.nodes()
        ]

        print(f"Number of categories not in tree: {len(nodes_not_in_tree)}")

        # Fill missing root categories with the item category itself
        df.loc[df["root_category"].isnull(), "root_category"] = df.loc[
            df["root_category"].isnull()
        ][self.config.fields_id["item_category"]]
        df["root_category"] = df["root_category"].astype("uint32")

        # Fill missing level with 0
        df["category_level"].fillna(0, inplace=True)

        # Distriution of the category level
        custom_hist_multiplot(
            data=df,
            columns=["category_level"],
            kde=False,
            features_kind="cat",
            cat_orient="h",
            stat="count",
            height_factor=0.7,
            savepath=Path(
                self.config.assets_path,
                f"category_level_hist.png",
            ),
            show=self.show_graphs,
            title="Distribution of the category level",
        )

        # Distribution of the root category count
        custom_hist_multiplot(
            data=df[["root_category"]].astype("str"),
            columns=["root_category"],
            kde=False,
            features_kind="cat",
            cat_orient="h",
            stat="count",
            height_factor=0.5,
            savepath=Path(
                self.config.assets_path,
                f"root_category_count_hist.png",
            ),
            show=self.show_graphs,
            title="Distribution of the root category count",
        )

        # Save the updated data
        df.to_parquet(
            Path(
                self.config.source_path,
                self.config.item_features_filenames["category"],
            )
        )

    def item_features_analysis(self) -> None:
        """
        Performs an analysis of the item features.
        """

        self.item_availability_analysis()

        node_levels, G = self.category_tree_analysis()

        self.item_category_analysis(
            G=G,
            category_levels=node_levels,
        )

    def analyze(self, log: bool = False) -> None:
        """
        Performs a full analysis on the training data.
        Args:
            log (bool):
                Whether to log the results. Defaults to False.
        """
        self.item_analysis()
        self.user_analysis()
        self.item_features_analysis()
        if log:
            self.log()
        logger.info(f"Finished EDA")
