import GA
import os
import numpy as np
import time
import pickle

directory = os.fsencode("Project/processed_data")
f_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    f_list.append(filename)

f_list.remove("aloi.pkl")  # TODO: remove line once Aloi preproc has been improved
for rem in ["mnist.pkl", "cifar10.pkl", "cifar100.pkl"]:
    f_list.remove(rem)

for f in f_list:
    perf = np.zeros(20)
    runtime = np.zeros(20)
    mse_wins = 0
    print("processing", f)
    xx, yy = GA.unpick(f)  # set /AML as working dir
    for i in range(20):
        print("###### file:", f, "; run:", i+1, "######")
        start_time = time.time()
        rs_obj = GA.RandomSearch(xx, yy)
        fit, loss = rs_obj.run()
        end_time = time.time()
        runtime[i] = end_time-start_time
        perf[i] = fit
        if loss == "mean_squared_error":
            mse_wins += 1
        res = {"perf": perf, "runtime": runtime, "mse_wins": mse_wins}
        with open("Project/rand_res/" + f, "wb") as file:
            pickle.dump(res, file)
