
import matplotlib
matplotlib.use('Agg')


from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sets import Set
from tqdm import tqdm

import matplotlib.pyplot as plt
import csv
import numpy as np
import sys


# Split into train and test sets (with same ratio)
def data_split(X, Y, pool_split=0.8):
	nb_classes = len(np.unique(Y))
	distr = {}
	for i in range(nb_classes):
		distr[i] = []
	for i in range(len(Y)):
		distr[Y[i]].append(i)
	X_bm_ret = []
	Y_bm_ret = []
	X_pm_ret = []
	Y_pm_ret = []
	# Calculate minimum number of points per class
	n_points = min([len(distr[i]) for i in distr.keys()])
	for key in distr.keys():
		st = np.random.choice(distr[key], n_points, replace=False)
		bm = st[:int(len(st)*pool_split)]
		pm = st[int(len(st)*pool_split):]
		X_bm_ret.append(X[bm])
		Y_bm_ret.append(Y[bm])
		X_pm_ret.append(X[pm])
		Y_pm_ret.append(Y[pm])
	X_bm_ret = np.concatenate(X_bm_ret)
	Y_bm_ret = np.concatenate(Y_bm_ret)
	X_pm_ret = np.concatenate(X_pm_ret)
	Y_pm_ret = np.concatenate(Y_pm_ret)
	return X_bm_ret, Y_bm_ret, X_pm_ret, Y_pm_ret


# Parse data
def parse_data(filename):
	X, Y = [], []
	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			X.append([ float(x) for x in row[2:]])
			Y.append(row[1])
	unique_labels = list(Set(Y))
	classes = {unique_labels[i]:i for i in range(len(unique_labels))}
	Y = list(map(lambda x: classes[x], Y))
	return np.array(X), np.array(Y)


# Perform k-fold cross validation
def kfoldSVM(X, Y, n_val, c, kernel, degree):
	kf = KFold(n_splits=n_val)
	kf.get_n_splits(X)
	score = 0
	for train_index, test_index in kf.split(X):
		svc = SVC(C=c, kernel=kernel, degree=degree)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		svc.fit(X_train, y_train)
		score += svc.score(X_test, y_test)
	score /= (1.0 * n_val)
	return score


def plotROC(svm, X, Y, prefix, configuration, n_classes=2):
	Y_pred = svm.predict(X)
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
    	fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_pred[:, i])
    	roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_pred.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(prefix + " " _+ configuration)
	plt.legend(loc="lower right")
	plt.show()


# Pick the SVM that performs best
def bestSVM(X, Y, n_val=4):
	C = [1e-5, 1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
	kernels = ['rbf', 'linear', 'poly']
	degrees = [2, 4, 6, 8, 10]
	xtr, ytr, xte, yte = data_split(X, Y, 0.8)
	scores = {}
	for c in tqdm(C):
		for kernel in kernels:
			for degree in degrees:
				configuration = str(c) + ":" + kernel + ":" + str(degree)
				scores[configuration] = kfoldSVM(xtr, ytr, n_val, c, kernel, degree)
	best_config = max(scores, key=scores.get)
	c, kernel, degree = best_config.split(":")
	svc = SVC(C=float(c), kernel=kernel, degree=int(degree))
	svc.fit(xtr, ytr)
	plotROC(svc, xtr, ytr, "train", best_config, 2)
	score = svc.score(xte, yte)
	plotROC(svc, xte, yte, "test", best_config, 2)
	return score


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python " + sys.argv[0] + " <filename>")
		exit()
	X, Y = parse_data(sys.argv[1])
	print(bestSVM(X, Y, 4))
