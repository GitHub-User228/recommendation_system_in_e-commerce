import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Literal, List, Dict

from scripts import logger
from scripts.utils import get_bins


def custom_hist_multiplot(
    data: pd.DataFrame | Dict[str, pd.DataFrame],
    columns: List[str] | None = None,
    hue: str | None = None,
    hue_order: list | None = None,
    title: str | None = None,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    kde: bool = True,
    features_kind: Literal["num", "cat"] = "num",
    cat_orient: Literal["v", "h"] = "v",
    savepath: Path | None = None,
    width_factor: float = 1,
    height_factor: float = 1,
    title_fontsize: int = 16,
    stat: Literal["density", "count"] = "density",
    show: bool = True,
) -> None:
    """
    Generates a multi-panel histogram plot for the given data and
    columns. Optionally saves the figure to a file.

    Parameters:
        data (pd.DataFrame | Dict[str, pd.DataFrame]):
            The input data. If a dictionary is provided, the keys are
            columns and the values are the corresponding dataframes.
            If a single dataframe is provided, the `columns` parameter
            should be provided.
        columns (List[str] | None, optional):
            The columns to plot histograms for. If None, `data` should be
            a dictionary with column names as keys and dataframes as values.
        hue (str | None, optional):
            The column to use for grouping the data. Defaults to None.
        hue_order (list | None, optional):
            The order of the hue values to display. Defaults to None.
        title (str | None, optional):
            The title of the plot. Defaults to None.
        features_kind (Literal["num", "cat"], optional):
            Whether the columns are numeric or categorical. Defaults
            to "num".
        cat_orient (Literal["v", "h"], optional):
            The orientation of the categorical histograms, vertical or
            horizontal. Defaults to "v".
        savepath (Path | None, optional):
            The path to save the plot to. Defaults to None.
        show (bool, optional):
            Whether to show the plot. Defaults to True.
    """
    if isinstance(data, dict):
        columns = list(data.keys())
    if features_kind == "num":
        fig, axs = plt.subplots(
            len(columns),
            figsize=(15 * width_factor, 4 * len(columns) * height_factor),
        )
        if len(columns) == 1:
            axs = [axs]
        if title:
            axs[0].set_title(title, fontsize=title_fontsize)
        for i, col in enumerate(columns):
            cols = [col] if hue is None else [col, hue]
            if isinstance(data, dict):
                data2 = data[col][cols].dropna()
            else:
                data2 = data[cols].dropna()
            sns.histplot(
                data=data2,
                x=col,
                ax=axs[i],
                hue=hue,
                hue_order=hue_order,
                common_norm=False,
                fill=True,
                kde=kde,
                alpha=0.6,
                stat=stat,
                bins=get_bins(len(data2)),
            )
            axs[i].set_xscale(xscale)
            axs[i].set_yscale(yscale)

            if hue:
                sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
    elif features_kind == "cat":
        if cat_orient == "h":
            nunique = sum([data[col].nunique() for col in columns])
            ratios = [data[col].nunique() / nunique for col in columns]
            fig, axs = plt.subplots(
                len(columns),
                figsize=(10 * width_factor, 0.55 * nunique * height_factor),
                gridspec_kw={"height_ratios": ratios},
            )
        elif cat_orient == "v":
            fig, axs = plt.subplots(
                len(columns),
                figsize=(15 * width_factor, 4 * len(columns) * height_factor),
            )
        if len(columns) == 1:
            axs = [axs]
        if title:
            fig.suptitle(title, fontsize=title_fontsize)
        for it, col in enumerate(columns):
            cols = [col] if hue is None else [col, hue]
            data2 = data[cols].dropna()
            x = col if cat_orient == "v" else None
            y = col if cat_orient == "h" else None
            sns.histplot(
                data=data2,
                x=x,
                y=y,
                discrete=True,
                hue=hue,
                hue_order=hue_order,
                common_norm=False,
                multiple="dodge",
                stat=stat,
                shrink=0.8,
                legend="full",
                ax=axs[it],
            )
            if cat_orient == "h":
                axs[it].set_ylabel("")
                axs[it].set_title(y)
                # if it < len(columns) - 1:
                axs[it].set_xlabel("")
                axs[it].set_yticks(sorted(data2[col].unique()))
                axs[it].grid(axis="y", linestyle="")
                # axs[it].set_ylim(data2[col].min() - 0.5, data2[col].max() + 0.5)
            elif cat_orient == "v":
                axs[it].set_xticks(sorted(data2[col].unique()))
                axs[it].grid(axis="x", linestyle="")
                # axs[it].set_xlim(data2[col].min() - 1, data2[col].max() + 1)
            if hue:
                sns.move_legend(axs[it], "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if show:
        plt.show()
    if savepath:
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        logger.info(f"custom_hist_multiplot saved to {savepath}")
