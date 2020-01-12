import pickle
from matplotlib import pyplot as plt
import numpy as np
import warnings

f_list = ['isolet', 'letter', 'sensorless', 'year', 'boston', 'ccpp', 'forest',
          'phsioco', 'ctslice', 'cifar100', 'mnist', 'crowdflower', 'imdb']
f_cat = 'ccccrrrrriiss'


def load_res(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def avg_ga(file_path, per_run=False):
    raw = load_res(file_path)
    runs_r = raw["runtime"] != 0
    runs_p = raw["perf"][:, 0, 0] != 0
    n_runs = np.sum(runs_r)
    perf = raw["perf"][runs_p]
    perf[perf == np.inf] = np.NaN  # replace inf with a NaN to enable calculating the mean
    avg_perf = np.nanmean(perf, axis=0)
    best_perf = perf[:, 1:3, :]
    best_perf = np.nanmin(best_perf, axis=(1, 2))
    mse_win = raw["mse_wins"]/n_runs
    mse_last = raw["mse_last"]/n_runs
    runtime = raw["runtime"][runs_r]
    avg_runtime = np.mean(runtime)
    assert len(runtime) == len(perf)
    if per_run:
        return perf, runtime, n_runs, mse_win, mse_last, best_perf
    else:
        return avg_perf, avg_runtime, n_runs, mse_win, mse_last, np.mean(best_perf)


def rand_res(file_path, n=20, per_run=False):
    raw = load_res(file_path)
    runs = raw["runtime"] != 0
    runs = runs[0:n]  # read all runs by default (20), but fewer runs can be selected by user
    n_runs = np.sum(runs)
    perf = raw["perf"][0:n][runs]
    avg_perf = np.mean(perf)
    best_perf = np.min(perf)
    runtime = raw["runtime"][0:n][runs]
    avg_runtime = np.mean(runtime)
    mse_win = raw["mse_wins"]/n_runs
    if per_run:
        return perf, runtime, n_runs, mse_win, best_perf
    else:
        return avg_perf, avg_runtime, n_runs, mse_win, best_perf


def plot_gen(file_name, save_file=None):
    perf, _, n1, _, _, best_ga = avg_ga("Project/GA_res/"+file_name+".pkl")
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.set_size_inches(12.8, 4.8)
    if file_name == "year":
        fig1.suptitle("Convergence Plot for Year Prediction")
    elif file_name == "forest":
        fig1.suptitle("Convergence Plot for Forest Fire")

    ax1.plot(perf[0], label="Proportion of MSE", ls=":")
    ax1.plot(perf[1], label="CCE Fitness")
    ax1.plot(perf[2], label="MSE Fitness")
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness/Proportion of MSE in Population")
    ax1.legend(prop={'size': 10})

    perf2, _, n2, _, _, best_ga = avg_ga("Project/GA_adjusted_newres/adjusted_" + file_name + "_results.pkl")
    ax2.plot(perf2[0], label="Proportion of MSE", ls=":")
    ax2.plot(perf2[1], label="CCE Fitness")
    ax2.plot(perf2[2], label="MSE Fitness")
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness/Proportion of MSE in Population")
    ax2.legend(prop={'size': 10})
    print("n1", n1, "n_adj", n2)

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def plot_adj(file_name, save_file=None, best=False):
    perf, _, n_ga, _, _, best_ga = avg_ga("Project/GA_adjusted_newres/adjusted_"+file_name+"_results.pkl")
    avg_rand, _, n_rand, _, _ = rand_res("Project/rand_res/"+file_name+".pkl", n_ga)
    if n_rand < n_ga:
        warnings.warn("unfair comparison: random had fewer runs")
        print("n_ga", n_ga, "n_rand", n_rand)
    fig1, ax1 = plt.subplots()
    ax1.plot(perf[0], label="Proportion of MSE")
    ax1.plot(perf[1], label="CEE Fitness")
    ax1.plot(perf[2], label="MSE Fitness")
    if best:
        ax1.axhline(avg_rand, ls=":", label="Best Random Fitness")
        ax1.axhline(best_ga, ls=":", c='r', label="Best GA Fitness")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness/Proportion of MSE in Population")
    ax1.legend(prop={'size': 10})
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def pct_wins(files, categories, save_file):
    with open(save_file, "w") as save:
        save.write("Data Set,Cat,API,APIa,Random,API,APIa,Random,\n")
    for f, cat in zip(files, categories):
        _, ga_rt, _, ga_win, _, _ = avg_ga("Project/GA_res/"+f+".pkl")
        _, ga2_rt, _, ga2_win, _, _ = avg_ga("Project/GA_adjusted_newres/adjusted_"+f+"_results.pkl")
        _, ran_rt, _, ran_win, _ = rand_res("Project/rand_res/"+f+".pkl")
        if cat != "r":
            ga_win = 1 - ga_win
            ga2_win = 1 - ga2_win
            ran_win = 1 - ran_win
        save_str = ','.join([str(i) for i in (f, cat, ga_win, ga2_win, ran_win,
                                              ga_rt, ga2_rt, ran_rt)])
        with open(save_file, "a") as save:
            save.write(save_str+",\n")


plot_gen("forest", "Project/forest.png")
plot_gen("year", "Project/year.png")
# plot_adj("year")
# pct_wins(f_list, f_cat, "Project/results.csv")

if __name__ == 'norun':
    # examples
    fig1, ax1 = plt.subplots()
    ax1.scatter(range(1, a_n + 1), a_plot["sum"], s=2, marker=1, label="AverageRank")
    ax1.scatter(range(1, g_n + 1), g_plot["sum"], s=2, marker=1, label="GreedyDefault")
    ax1.set_title("Sum of Max Performance per Dataset as Configs are added in Rank Order")
    ax1.set_xlabel("Number of Configs")
    ax1.set_ylabel("Sum of Max Performance per Dataset")
    ax1.legend()

    fig2, (ax2, ax3) = plt.subplots(2, 1, sharey=True)
    ax2.boxplot(g_plot["mps"], flierprops={"marker": "x", "markersize": 2})
    ax2.legend(["GreedyDefault"], loc=(0.01, 0.12))
    ax3.boxplot(a_plot["mps"][0:288:9], labels=range(1, 289, 9),
                flierprops={"marker": "x", "markersize": 2})
    ax3.legend(["AverageRank"], loc=(0.01, 0.12))
    ax3.set_xlabel("Number of Configs")
    fig2.text(0.06, 0.5, "Max Performance per Dataset", ha='center', va='center', rotation='vertical')
    fig2.suptitle("Distribution of Max Performance per Dataset")

    plt.show()