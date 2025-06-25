# Prepare some boilerplate code and read the pickle
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import pint

# The following lines are required because conllovia uses pint library to deal
# with units, and defines some custom units that need to be registered before
# reading the pickle
from cloudmodel.unified.units import ureg

pint.set_application_registry(ureg)


# Utility function to plot the workload
def plot_workloads(initial_hour=9, duration=3600):
    app0_wl = pd.read_csv(
        "additional_files/data/wl_static_1s.csv",
        header=None,
        skiprows=range(0, initial_hour * 3600),
        nrows=duration,
    )
    app1_wl = pd.read_csv(
        "additional_files/data/wl_unpredictable_1s.csv",
        header=None,
        skiprows=range(0, initial_hour * 3600),
        nrows=duration,
    )
    plt.plot(app0_wl[0], label="app0", linewidth=0.5)
    plt.plot(app1_wl[0], label="app1", linewidth=0.5)
    plt.xlabel("seconds")
    plt.ylabel("rps")
    plt.title("rps per application")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# Utility functions to present the problem parameters and solution as tables
def get_allocation_df(allocation):
    def extract_container_data(container, num_replicas):
        return {
            "Instance Class": container.vm.ic.name,
            "VM name": f"vm{int(container.vm.name().split('-')[-1]):02d}",
            "Container Class": container.cc.name,
            r"# repl": int(num_replicas),
        }

    alloc = [
        extract_container_data(container, replicas)
        for container, replicas in allocation.containers.items()
        if replicas > 0
    ]
    df = pd.DataFrame(alloc)
    return df.set_index(["Instance Class", "VM name", "Container Class"])


def workload_summary(workloads):
    data = []
    for app, wl in workloads.items():
        rps = wl.num_reqs.m / wl.time_slot_size.to("s").m
        reqs = wl.num_reqs.m
        data.append({"Application": app.name, "Total requests": reqs, "rps": rps})
    return pd.DataFrame(data)


def allocated_vms(alloc_df):
    df_vms = (
        alloc_df.reset_index()
        .groupby("Instance Class")["VM name"]
        .nunique()
        .reset_index(name="VM count")
    )
    return df_vms


def allocated_ccs(alloc_df):
    df_ccs = (
        alloc_df.reset_index()
        .groupby(["Container Class", "Instance Class"])["# repl"]
        .sum()
        .reset_index(name="# replicas")
        .sort_values(by="Container Class", key=lambda x: x.str[-1])
    )
    df_ccs["app"] = df_ccs["Container Class"].apply(lambda x: x[-4:])
    return df_ccs.set_index(["app", "Container Class", "Instance Class"])


def get_performances_df(perfs):
    df = pd.DataFrame(
        [(ic.name, cc.name, v.m) for ((ic, cc), v) in perfs.items()],
        columns=["Instance Class", "Container Class", "performance"],
    )
    # Put it as a matrix for easier reading
    df_pivot = (
        df.pivot(
            index="Instance Class", columns="Container Class", values="performance"
        )
        / 3600
    )
    return df_pivot


def read_solution(filename):
    with open(filename, "rb") as f:
        solution = pickle.load(f)
    return solution
