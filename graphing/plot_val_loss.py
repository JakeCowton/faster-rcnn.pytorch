from os import path, listdir

from pandas import read_csv
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt



def read_data(fps):
    return {k: read_csv(fp) for k, fp in fps.items()}

def main(fps):
    dfs = read_data(fps)

    fig = plt.figure(figsize=(20,10))
    for i, (name, df) in enumerate(dfs.items()):
        val = df[df.stage == "val"]
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(val.epoch, val.score, c="xkcd:denim blue", label="Val")
        ax.plot(val.epoch, [val.score.max()]*len(val), c="xkcd:medium green",
                label=str(round(val.score.max(), 3)))

        # If there's no test, move on
        if len(df) != len(val):
            test = df[df.stage == "test"]
            ax.scatter(test.epoch, test.score, c="xkcd:pale red",
                       label=f"Test {round(test.score.max(), 3)}")

        ax.legend()
        ax.set_title(name)
        ax.set_xticks(list(range(df.epoch.min(), df.epoch.max()+1)))
        ax.set_ylim([0,1])

    plt.savefig("/home/jake/performance.png")


if __name__ == "__main__":
    root_dir = "/home/jake/src/faster-rcnn.pytorch/logs/"
    fps = {folder: path.join(root_dir, folder, "performance.csv")\
            for folder in sorted(listdir(root_dir))}
    main(fps)
