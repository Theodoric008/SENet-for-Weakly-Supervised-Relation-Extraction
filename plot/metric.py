import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('Agg')

# fill in pkl paths in tuple
pkls = (
    '',
    '',

       )
for item in pkls:
    print("running : ", item)
    outpacks = pkl.load(open(item, "rb"))
    results = outpacks["result"]
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

    order = np.argsort(-allprob)

    print('P@100:')
    top100 = order[:100]
    correct_num_100 = 0.0
    for i in top100:
        if allans[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / 100.0)
    ans1 = correct_num_100 / 100.0

    print('P@200:')
    top200 = order[:200]
    correct_num_200 = 0.0
    for i in top200:
        if allans[i] == 1:
            correct_num_200 += 1.0
    print(correct_num_200 / 200.0)
    ans2 = correct_num_200 / 200.0

    print('P@300:')
    top300 = order[:300]
    correct_num_300 = 0.0
    for i in top300:
        if allans[i] == 1:
            correct_num_300 += 1.0
    print(correct_num_300 / 300.0)
    ans3 = correct_num_300 / 300.0

    print((ans1 + ans2 + ans3) / 3.0)
    print('------')



