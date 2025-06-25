# Code for plotting
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def filter_data(data_df, experiment=None):
    if experiment is not None:
        data_df = data_df[data_df["experiment"] == experiment]
    return data_df


def select_experiment_df(data, distribution, termination, lb, queue_size):
    selected_experiment_name = (
        f"{distribution}_" f"{termination}_" f"{lb}_" f"{queue_size}"
    )
    selected_data = filter_data(data, experiment=selected_experiment_name)
    return selected_data


def plot_workloads(data, case: str, aggregate_time="1s", ax=None):
    selected_data = select_experiment_df(data, case, "hard", "SWRR", "q0")
    _df_injected_per_second = selected_data.set_index("injected")
    requests_per_second = (
        _df_injected_per_second.groupby("app")["app"]
        .resample("1s")
        .size()
        .reset_index(name="request_count")
    )

    _requests_per_second_indexed = requests_per_second.set_index("injected")

    # Group per app
    max_requests_per_minute = (
        _requests_per_second_indexed.groupby("app")["request_count"]
        .resample(aggregate_time)
        .mean()
        .reset_index(name="requests")
    )

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_standalone = True
    else:
        fig = ax.figure
        show_standalone = False
    sns.lineplot(
        data=max_requests_per_minute,
        x="injected",
        y="requests",
        hue="app",
        linewidth=1.5,
        alpha=0.8,
        ax=ax,
    )

    # Show only HH:MM in the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.set_title(
        f"Average workload for the '{case}' injector, window size of {aggregate_time}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Requests")
    ax.grid(True, linestyle="--", alpha=0.6)
    if show_standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_response_time(
    data,
    distribution: str,
    termination: str,
    lb: str,
    queue_size: str,
    app: Optional[str] = None,
    grouper: Optional[str] = None,
    avg_window: str = "60s",
    ax=None,
):
    """
    This function selects one experiment data and plots a line chart
    with error bars showing the average response time per time bin,
    grouped by the specified grouper (app or container).
    """
    selected_data = select_experiment_df(
        data, distribution, termination, lb, queue_size
    )
    if app is not None:
        selected_data = selected_data[selected_data["app"] == app]
    if grouper is None:
        grouper = "app"
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_standalone = True
    df_rt = selected_data.copy().query("finished==True")
    df_rt["time_bin"] = df_rt["injected"].dt.floor(avg_window)

    sns.lineplot(
        data=df_rt,
        x="time_bin",
        y="response_time",
        hue=grouper,
        linewidth=2,
        marker="o",
        alpha=0.5,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.ticklabel_format(style="plain", axis="y")
    if grouper == "app":
        to_title = "per application"
        legend_title = "Application"
    else:
        to_title = "per container"
        legend_title = "Container type"
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    title = f"Average response time {to_title} for scenario <{distribution}, {termination}, {lb}, {queue_size}>"
    ax.set_title(title)
    ax.set_ylabel(f"Average Response Time ({avg_window} window)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax


def plot_requests_success(
    data,
    distribution: str,
    termination: str,
    lb: str,
    queue_size: str,
    app: Optional[str] = None,
    grouper: Optional[str] = None,
    avg_window: str = "60s",
    ax=None,
):
    """
    This function selects one experiment data and plots a stacked bar chart
    showing the number of completed and lost requests per time bin,
    grouped by the specified grouper (app or container)
    """

    selected_data = select_experiment_df(
        data, distribution, termination, lb, queue_size
    )
    if app is not None:
        selected_data = selected_data[selected_data["app"] == app]
    if grouper is None:
        grouper = "app"

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 7))
        show_standalone = True
    else:
        fig = ax.figure
        show_standalone = False

    # Group per time bin and grouper
    df_agg = selected_data.copy()
    df_agg["time_bin"] = df_agg["injected"].dt.floor(avg_window)
    requests_counts = df_agg.pivot_table(
        index="time_bin",
        columns=[grouper, "finished"],
        values="experiment",
        aggfunc="count",
        fill_value=0,
    )

    if requests_counts.empty:
        ax.text(0.5, 0.5, "No data for this selection", ha="center", va="center")
        return ax

    requests_counts = requests_counts.sort_index(
        axis=1, level=["finished", grouper], ascending=[False, True]
    )

    # Custom colours and legend.
    # Completed requests will be green, lost requests will be red.
    unique_groups = sorted(requests_counts.columns.get_level_values(grouper).unique())
    n_groups = len(unique_groups)
    red_palette = sns.color_palette("Reds_d", n_groups)
    green_palette = sns.color_palette("Greens_d", n_groups)
    group_to_red = dict(zip(unique_groups, red_palette))
    group_to_green = dict(zip(unique_groups, green_palette))

    # Bars are drawn using low-level matplotlib API, instead of pandas.bar
    # to allow for custom stacking and xlabel tick formatting.
    width_in_seconds = pd.to_timedelta(avg_window).total_seconds()
    width_in_days = (width_in_seconds / (24 * 3600)) * 0.8

    # Lógica de apilado
    bottom_executed = pd.Series(0, index=requests_counts.index, dtype="float64")
    executed_cols = [col for col in requests_counts.columns if col[1] is True]
    bottom_lost_start = (
        requests_counts[executed_cols].sum(axis=1)
        if executed_cols
        else pd.Series(0, index=requests_counts.index, dtype="float64")
    )
    bottom_lost = bottom_lost_start.copy()

    for (group, status), series in requests_counts.items():
        if status:  # Executed
            color = group_to_green[group]
            ax.bar(
                series.index,
                series.values,
                bottom=bottom_executed.values,
                width=width_in_days,
                color=color,
                align="edge",
            )
            bottom_executed += series.values
        else:  # Lost
            color = group_to_red[group]
            ax.bar(
                series.index,
                series.values,
                bottom=bottom_lost.values,
                width=width_in_days,
                color=color,
                align="edge",
            )
            bottom_lost += series.values

    title = f"Executed vs Lost Requests per {grouper} for scenario <{distribution}, {termination}, {lb}, {queue_size}>"
    ax.set_title(title)
    ax.set_ylabel("Number of Requests")
    ax.set_xlabel("Time")

    # --- Formateo del Eje X ---
    ## ax.xaxis.set_major_locator(mdates.MinuteLocator(by_minute=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # fig.autofmt_xdate()

    # --- Creación de Leyenda Manual ---
    legend_handles = []
    for group in unique_groups:
        legend_handles.append(
            Patch(color=group_to_green[group], label=f"{group} | Completed")
        )
    for group in unique_groups:
        legend_handles.append(Patch(color=group_to_red[group], label=f"{group} | Lost"))

    ax.legend(
        handles=legend_handles,
        title=f"{grouper} | Status",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if show_standalone:
        plt.tight_layout()
        plt.show()

    return ax
