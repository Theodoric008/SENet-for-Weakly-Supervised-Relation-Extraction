import numpy as np
import pickle as pkl
import os
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')

# save test result here
folder_path = 'temp/'
global_max = 0
global_max_pkl = ''
to_test = os.listdir(folder_path)

def sort_file_name(file_list):
    res = []
    save_dict = {}
    for file in file_list:
        num = file.split('.')[0]
        num = num.split('_')[-1]
        num = int(num)
        save_dict[num] = file
    for key in sorted(save_dict.keys()):
        res.append(save_dict[key])
    return res

to_test = sort_file_name(to_test)

for pkls in to_test:
    print("testing :", pkls)
    pkls = os.path.join(folder_path, pkls)
    outpacks = pkl.load(open(pkls, "rb"))

    results = outpacks["result"]

    correct = 0
    allprob = []
    allans = []
    for i in range(len(results)):
        ans = [0]*53
        ans[results[i][1]] = 1
        ans = ans[1:]
        allans.append(ans)
        allprob.append(results[i][4])

    allans = np.reshape(np.array(allans), (-1))
    allprob = np.reshape(np.array(allprob), (-1))

    curr_pkl_score = average_precision_score(allans, allprob)

    print("average_precision_score for curr pkl is : ", curr_pkl_score)
    if curr_pkl_score > global_max:
        global_max = curr_pkl_score
        global_max_pkl = pkls

print("-----done-----")
print("global max is:", global_max)
print("global max pkl is:", global_max_pkl)
