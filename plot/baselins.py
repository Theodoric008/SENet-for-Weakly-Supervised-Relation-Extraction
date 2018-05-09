import pickle as pkl
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import embed


# filename = ['CNN+ATT','Hoffmann','MIMLRE','Mintz','PCNN+ATT']
filename = ['CNN+ATT', 'PCNN+ATT']
color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
line_style = ['--', '-.', ':', '-']
marker = ['.', ',', 'o', 'v']
for i in range(len(filename)):
	precision = np.load('./data/'+filename[i]+'_precision.npy')
	recall  = np.load('./data/'+filename[i]+'_recall.npy')
	plt.plot(recall,precision,color = color[i], linestyle=line_style[i], lw=2.5,label = filename[i], marker=marker[i])


# ResCNN 9
# Fill in the path of the model that you wanna compare with
pkls = ''
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
	ans = [0] * 53
	ans[results[i][1]] = 1
	ans = ans[1:]
	allans.append(ans)
	allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)
print(curr_pkl_score)
precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color=color[2], linestyle=':', lw=1.8, label='ResCNN-9')



# BLSTM + ATT
pkls = ''
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
	ans = [0] * 53
	ans[results[i][1]] = 1
	ans = ans[1:]
	allans.append(ans)
	allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)
print(curr_pkl_score)
precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color='orchid', linestyle='--', lw=1.5, label='BLSTM+ATT')


# BiGRU + 2ATT
pkls = ''
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
	ans = [0] * 53
	ans[results[i][1]] = 1
	ans = ans[1:]
	allans.append(ans)
	allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)
print(curr_pkl_score)
precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color='teal', linestyle='-.', lw=2.2, label='BGRU+2ATT')


# my soa
pkls = ''
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
	ans = [0] * 53
	ans[results[i][1]] = 1
	ans = ans[1:]
	allans.append(ans)
	allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)
print(curr_pkl_score)
precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color='cornflowerblue', linestyle='-', lw=2, label='SE-ResNet-D')


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 0.6])
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('baselines_soa.eps')
plt.savefig('baselines_soa.png')

