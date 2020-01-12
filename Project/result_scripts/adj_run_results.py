import GA_adjusted
import os
import numpy as np
import time
import pickle

directory = os.fsencode("Project/processed_data/")
f_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    f_list.append(filename)

f_list.remove("aloi.pkl")  # TODO: remove line once Aloi preproc has been improved
for rem in ["mnist.pkl", "cifar10.pkl", "cifar100.pkl"]:
    f_list.remove(rem)

for f in f_list:
    perf = np.zeros((5, 3, 11), dtype=np.float)
    runtime = np.zeros(5)
    mse_wins = 0
    mse_wins_last_gen = 0
    print("processing", f)
    xx, yy = GA_adjusted.unpick(f)  # set /AML as working dir
    for i in range(5):
        print("###### file:", f, "; run:", i+1, "######")
        start_time = time.time()
        ga_adj_obj = GA_adjusted.GA_adjusted(xx, yy, pop_size=50, max_generations=10)
        _ = ga_adj_obj.run()
        end_time = time.time()
        runtime[i] = end_time-start_time
        perf[i, ...] = np.array([ga_adj_obj.hist["mse_pct"],
                                 ga_adj_obj.hist["ce_fit"],
                                 ga_adj_obj.hist["mse_fit"]])
        if min(ga_adj_obj.hist["mse_fit"]) < min(ga_adj_obj.hist["ce_fit"]):
            mse_wins += 1
        if ga_adj_obj.hist["mse_fit"][-1] < ga_adj_obj.hist["ce_fit"][-1]:
            mse_wins_last_gen += 1
        res = {"perf": perf, "runtime": runtime, "mse_wins": mse_wins, "mse_last": mse_wins_last_gen}
        with open("Project/GA_adjusted_newres/" + f, "wb") as file:
            pickle.dump(res, file)
