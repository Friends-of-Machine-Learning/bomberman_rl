import pandas as pd
import plotly.express as px

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

WIDTH = 800
HEIGHT = 600


def generate_reward_plot(out_dir, reward_file):
    rewards = []
    with open(reward_file, "r") as file:
        for line in file:
            rewards.append(float(line))

    df_rewards = pd.DataFrame(data=rewards, columns=["rewards"])

    df_rewards = pd.DataFrame(data=rewards, columns=["rewards"])
    fig = px.line(
        df_rewards,
        title="Rewards per Round",
        labels={"value": "Rewards", "index": "Round"},
        template="presentation",
        width=WIDTH,
        height=HEIGHT,
    )
    fig.write_image(out_dir + reward_file.split("/")[-2] + ".svg")


def generate_mean_plot(out_dir, mean_file):
    means = []
    with open(mean_file, "r") as file:
        for line in file:
            line = [float(num) for num in line.split(" ")]
            means.append(line)

    df_means = pd.DataFrame(data=means, columns=ACTIONS)

    fig = px.line(
        df_means,
        title="Mean return per Action",
        labels={"value": "Mean return", "index": "Round"},
        template="presentation",
        width=WIDTH,
        height=HEIGHT,
    )

    fig.write_image(out_dir + mean_file.split("/")[-2] + ".svg")


if __name__ == "__main__":
    # create plots folder if it does not yet exist.
    import os

    if not os.path.exists("plots"):
        os.makedirs("plots")

    # generate mean plots for every means.txt file in all subdirectories.
    import glob

    for mean_file in glob.glob("**/means.txt", recursive=True):
        generate_mean_plot("plots/", mean_file)

    # generate reward plots for every game_rewards.txt file in all subdirectories.
    for reward_file in glob.glob("**/game_rewards.txt", recursive=True):
        generate_reward_plot("plots/", reward_file)
