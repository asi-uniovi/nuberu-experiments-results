# Code for plotting
from typing import List, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def filter_data(data_df, experiment=None):
    if experiment is not None:
        data_df = data_df[data_df["experiment"] == experiment]
    return data_df


def select_experiment_df(data, distribution, termination, lb, queue_size, lat="l0"):
    selected_experiment_name = f"{distribution}_{termination}_{lb}_{queue_size}_{lat}"
    selected_data = filter_data(data, experiment=selected_experiment_name)
    return selected_data


def plot_workloads(data, case: str, aggregate_time="1s", latency: str = "l0", ax=None):
    selected_data = select_experiment_df(data, case, "hard", "SWRR", "q0", latency)
    if selected_data.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            show_standalone = True
        else:
            show_standalone = False
        ax.text(0.5, 0.5, "No data for this selection", ha="center", va="center")
        if show_standalone:
            plt.tight_layout()
            plt.show()
        return ax

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
    hue_order = sorted(max_requests_per_minute["app"].unique())
    sns.lineplot(
        data=max_requests_per_minute,
        x="injected",
        y="requests",
        hue="app",
        hue_order=hue_order,
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
    latency: str = "l0",
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
        data, distribution, termination, lb, queue_size, latency
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

    hue_order = sorted(df_rt[grouper].unique())
    sns.lineplot(
        data=df_rt,
        x="time_bin",
        y="response_time",
        hue=grouper,
        hue_order=hue_order,
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
    title = f"Average response time {to_title} for scenario <{distribution}, {termination}, {lb}, q{queue_size}, {latency}>"
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
    latency: str = "l0",
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
        data, distribution, termination, lb, queue_size, latency
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

    title = f"Executed vs Lost Requests per {grouper} for scenario <{distribution}, {termination}, {lb}, q{queue_size}, {latency}>"
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


def plot_response_time_cdf(
    data,
    distribution: str,
    termination: str,
    lb: str,
    queue_size: str,
    latency: Union[str, List[str]] = "l0",
    app: Optional[str] = None,
    grouper: Optional[str] = None,
    xlim: Optional[tuple] = None,
    log_scale: bool = False,
    percentiles: Optional[List[float]] = None,
    ax=None,
):
    """
    Plots the Cumulative Distribution Function (CDF) of the response time,
    grouped by the specified grouper (app or container) and optionally by latency.
    """
    latencies = [latency] if isinstance(latency, str) else latency

    all_data = []
    for lat in latencies:
        df_lat = select_experiment_df(
            data, distribution, termination, lb, queue_size, lat
        ).copy()
        if not df_lat.empty:
            df_lat["latency_val"] = lat
            all_data.append(df_lat)

    if not all_data:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "No data found for these parameters", ha="center", va="center"
        )
        return ax

    selected_data = pd.concat(all_data)

    if app is not None:
        selected_data = selected_data[selected_data["app"] == app]
    if grouper is None:
        grouper = "app"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_standalone = True
    else:
        fig = ax.figure
        show_standalone = False

    df_rt = selected_data.query("finished==True").copy()

    if df_rt.empty:
        ax.text(0.5, 0.5, "No data for this selection", ha="center", va="center")
        return ax

    # If multiple latencies are provided, we create a composite label
    if len(latencies) > 1:
        df_rt["plot_label"] = (
            df_rt[grouper] + " (lat=" + df_rt["latency_val"].str[1:] + ")"
        )
        hue_col = "plot_label"
    else:
        hue_col = grouper

    hue_order = sorted(df_rt[hue_col].unique())

    # Plot
    sns.ecdfplot(
        data=df_rt,
        x="response_time",
        hue=hue_col,
        hue_order=hue_order,
        ax=ax,
        alpha=0.8,
    )

    if percentiles:
        # Get the colors from the palette to match hue_order
        palette = sns.color_palette(n_colors=len(hue_order))
        for p in percentiles:
            ax.axhline(p, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
            for i, label in enumerate(hue_order):
                val = df_rt[df_rt[hue_col] == label]["response_time"].quantile(p)
                color = palette[i]
                ax.axvline(val, color=color, linestyle=":", alpha=0.5, linewidth=1)

                # Stagger text vertically to avoid overlap.
                # Second one (index 1) goes below to avoid getting cut at the top (Y=1.0)
                y_pos = p + 0.005 if i % 2 == 0 else p - 0.005
                va = "bottom" if i % 2 == 0 else "top"
                ax.text(
                    val,
                    y_pos,
                    f"{val:.3f}",
                    color=color,
                    fontsize=8,
                    va=va,
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                )

    if grouper == "app":
        to_title = "per application"
        legend_title = "Application"
    else:
        to_title = "per container"
        legend_title = "Container type"

    # Robust queue name for title
    q_name = queue_size if str(queue_size).startswith("q") else f"q{queue_size}"

    title = f"CDF of response time {to_title}\nScenario: <{distribution}, {termination}, {lb}, {q_name}, {latency}>"
    ax.set_title(title)
    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True, linestyle="--", alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    if log_scale:
        ax.set_xscale("log")

    ax.set_ylim(0, 1.05)
    # Move legend outside
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title=legend_title)

    # --- Annotation for app0 delta ---
    # Add a dimension line at P50 (y=0.5) to show the horizontal distance
    # between the two curves for "app0" (if applicable).
    if "app" in df_rt.columns and "response_time" in df_rt.columns:
        df_app0 = df_rt[df_rt["app"] == "app1"]
        if not df_app0.empty:
            # Check if we have two curves for app0.
            # The curves are distinguished by 'hue_col'.
            groups_app0 = df_app0[hue_col].unique()
            if len(groups_app0) == 2:
                # Calculate medians (P50)
                medians = df_app0.groupby(hue_col)["response_time"].median()
                if len(medians) == 2:
                    val1, val2 = medians.iloc[0], medians.iloc[1]
                    xmin, xmax = min(val1, val2), max(val1, val2)

                    # Draw arrows pointing inwards from the outside
                    # Left arrow: points to xmin from the left
                    ax.annotate(
                        "",
                        xy=(xmin, 0.5),
                        xytext=(-20, 0),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    )
                    # Right arrow: points to xmax from the right
                    ax.annotate(
                        "",
                        xy=(xmax, 0.5),
                        xytext=(20, 0),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    )

                    # Add text label centered
                    # Round delta to nearest 10ms
                    delta_ms = (xmax - xmin) * 1000
                    delta_ms_rounded = round(delta_ms / 10) * 10
                    ax.text(
                        (xmin + xmax) / 2,
                        0.53,
                        rf"$\Delta \approx {delta_ms_rounded:.0f}$ms",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        bbox=dict(
                            facecolor="white", alpha=0.8, edgecolor="none", pad=1
                        ),
                    )

    if show_standalone:
        plt.tight_layout()
        plt.show()

    return ax


def get_response_time_quantiles(
    data,
    distribution: str,
    termination: str,
    lb: str,
    queue_size: str,
    latency: Union[str, List[str]] = "l0",
    app: Optional[str] = None,
    grouper: Optional[str] = None,
    quantiles: List[float] = [0.5, 0.9, 0.95, 0.99],
):
    """
    Returns a DataFrame with the response time quantiles for the given scenario,
    useful for measuring the exact distance between CDF curves.
    """
    latencies = [latency] if isinstance(latency, str) else latency
    all_data = []
    for lat in latencies:
        df_lat = select_experiment_df(
            data, distribution, termination, lb, queue_size, lat
        ).copy()
        if not df_lat.empty:
            df_lat["latency_val"] = lat
            all_data.append(df_lat)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data)
    if app is not None:
        df = df[df["app"] == app]
    if grouper is None:
        grouper = "app"

    df_rt = df.query("finished==True").copy()

    if len(latencies) > 1:
        df_rt["group_label"] = df_rt[grouper] + " (" + df_rt["latency_val"] + ")"
        group_col = "group_label"
    else:
        group_col = grouper

    # Calculate quantiles per group
    results = []
    for group, group_df in df_rt.groupby(group_col):
        q_values = group_df["response_time"].quantile(quantiles)
        q_row = {"group": group}
        for q, val in zip(quantiles, q_values):
            q_row[f"P{int(q * 100)}"] = val
        results.append(q_row)

    return pd.DataFrame(results).set_index("group")
