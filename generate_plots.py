import glob
import json
import os

import pandas as pd
import plotly.express as px

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

WIDTH = 800
HEIGHT = 600

MIN_MAX_PLOT_ROUNDS = 25_000


def generate_reward_plot(out_dir, reward_file):
    rewards = []

    ln = 0

    with open(reward_file, "r") as file:
        for ln, line in enumerate(file):
            rewards.append(float(line))
            if ln > MIN_MAX_PLOT_ROUNDS:
                break

    if ln < MIN_MAX_PLOT_ROUNDS:
        return

    df_rewards = pd.DataFrame(data=rewards, columns=["rewards"])

    df_rewards = pd.DataFrame(data=rewards, columns=["rewards"])
    fig = px.line(
        df_rewards,
        title="",
        labels={"value": "Rewards", "index": "Round"},
        template="presentation",
        width=WIDTH,
        height=HEIGHT,
    )
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "right",
            "x": 1,
            "title_text": "",
        }
    )

    fig.write_image(out_dir + reward_file.split("/")[-2] + "_rewards.svg")


def generate_mean_plot(out_dir, mean_file):
    means = []
    ln = 0
    with open(mean_file, "r") as file:
        for ln, line in enumerate(file):
            line = [float(num) for num in line.split(" ")]
            means.append(line)
            if ln > MIN_MAX_PLOT_ROUNDS:
                break

    if ln < MIN_MAX_PLOT_ROUNDS:
        return

    df_means = pd.DataFrame(data=means, columns=ACTIONS)

    fig = px.line(
        df_means,
        title="",
        labels={"value": "Mean return", "index": "Round"},
        template="presentation",
        width=WIDTH,
        height=HEIGHT,
    )
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "right",
            "x": 1,
            "title_text": "",
        }
    )

    fig.write_image(out_dir + mean_file.split("/")[-2] + "_mean.svg")


def generate_qtable_plots(out_dir, qtable_file):
    means = []
    ln = 0
    with open(qtable_file, "r") as file:
        for ln, line in enumerate(file):
            line = [float(num) for num in line.split(" ")][1]
            means.append(line)
            if ln > MIN_MAX_PLOT_ROUNDS:
                break

    # if ln < MIN_MAX_PLOT_ROUNDS:
    #    return

    df = pd.DataFrame(data=means, columns=["Observed States"])

    fig = px.line(
        df,
        title="",
        labels={"value": "Count", "index": "Round"},
        template="presentation",
        width=WIDTH,
        height=HEIGHT,
    )
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "right",
            "x": 1,
            "title_text": "",
        }
    )

    fig.write_image(out_dir + qtable_file.split("/")[-2] + "_qtable.svg")


def generate_performance_comp_plot(out_dir, perf_file):
    results: dict
    with open(perf_file, "r") as file:
        results = json.load(file)

    by_agent = results["by_agent"]

    df = pd.DataFrame(by_agent)
    print(df)

    for _, row in df.iterrows():
        fig = px.bar(
            row,
            title="",
            labels={"value": "Count (sum of 1k rounds)", "index": "Agents"},
            template="presentation",
            width=WIDTH,
            height=HEIGHT,
            text="value",
        )
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1,
                "xanchor": "right",
                "x": 1,
                "title_text": "",
            },
            xaxis={"tickangle": 0, "tickfont": {"size": 12}},
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

        fig.write_image(
            out_dir + perf_file.split("/")[-1].split(".")[0] + f"performance_{_}.svg"
        )


def generate_comparison_plots(out_dir, perf_files):

    results: dict = {}

    for f in perf_files:
        with open(f, "r") as file:
            results.update(json.load(file)["by_agent"])

    df = pd.DataFrame(results)
    df = df[df.columns.drop(list(df.filter(regex="rule_based*")))]
    print(df)

    for _, row in df.iterrows():
        fig = px.bar(
            row,
            title="",
            labels={"value": "Count (sum of 1k rounds)", "index": "Agents"},
            template="presentation",
            width=WIDTH,
            height=HEIGHT,
            text="value",
        )
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1,
                "xanchor": "right",
                "x": 1,
                "title_text": "",
            },
            xaxis={"tickangle": 0, "tickfont": {"size": 12}},
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

        fig.write_image(out_dir + f"comparison_performance_{_}.svg")


if __name__ == "__main__":
    # create plots folder if it does not yet exist.

    if not os.path.exists("plots"):
        os.makedirs("plots")

    # generate mean plots for every means.txt file in all subdirectories.
    for mean_file in glob.glob("**/plot_agent/**/means.txt", recursive=True):
        generate_mean_plot("plots/", mean_file)

    # generate reward plots for every game_rewards.txt file in all subdirectories.
    for reward_file in glob.glob("**/plot_agent/**/game_rewards.txt", recursive=True):
        generate_reward_plot("plots/", reward_file)

    for qtable_file in glob.glob(
        "**/plot_agent/**/qtable_sparseness.txt", recursive=True
    ):
        generate_qtable_plots("plots/", qtable_file)

    for f in glob.glob("**/results/*.json", recursive=True):
        generate_performance_comp_plot("plots/", f)

    generate_comparison_plots("plots/", glob.glob("**/results/*.json", recursive=True))
