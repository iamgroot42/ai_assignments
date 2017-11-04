import matplotlib
matplotlib.use('Agg')


from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sets import Set
from tqdm import tqdm

import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
from keras.utils import np_utils


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
def kfoldSVM(X, Y, n_val, c, kernel):
	kf = KFold(n_folds=n_val)
	kf.get_n_splits(X)
	score = 0
	trscores, tescores = [], []
	for train_index, test_index in kf.split(X):
		svc = SVC(C=c, kernel=kernel)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		svc.fit(X_train, y_train)
		trscores.append(svc.score(X_train, y_train))
		score += svc.score(X_test, y_test)
		tescores.append(svc.score(X_test, y_test))
	score /= (1.0 * n_val)
	# Plot scores across folds
	plt.figure()
        plt.plot(np.arange(1,n_val+1), trscores, color='darkorange', lw=2, label='Train accuracy')
        plt.plot(np.arange(1,n_val+1), tescores, color='navy', lw=2, label='Test accuracy', linestyle='--')
        plt.ylim([0.0, 1.05])
        plt.xlabel('Fold')
        plt.ylabel('Accuracy on folds')
        plt.title(str(c)+":"+kernel)
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(str(c) + ":" + kernel + ".png")
	plt.clf()
	return score


def plotROC(svm, X, Y, prefix, configuration, n_classes=2):
	Y_pred = svm.predict(X)
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y = np_utils.to_categorical(Y, n_classes)
	Y_pred = np_utils.to_categorical(Y_pred, n_classes)
	for i in range(n_classes):
    		fpr[i+1], tpr[i+1], _ = roc_curve(Y[:, i], Y_pred[:, i])
	    	roc_auc[i+1] = auc(fpr[i+1], tpr[i+1])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_pred.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(prefix + " " + configuration)
	plt.legend(loc="lower right")
	plt.show()
	plt.savefig(prefix + " " + configuration + ".png")


# Pick the SVM that performs best
def bestSVM(X, Y, n_val=4):
	C = [1e-5, 1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, np.inf]
	kernels = ['rbf', 'linear']
	xtr, ytr, xte, yte = data_split(X, Y, 0.8)
	scores = {}
	for c in tqdm(C):
		for kernel in kernels:
			configuration = str(c) + ":" + kernel
			scores[configuration] = kfoldSVM(xtr, ytr, n_val, c, kernel)
	best_config = max(scores, key=scores.get)
	c, kernel = best_config.split(":")
	svc = SVC(C=float(c), kernel=kernel)
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
	print("For the best configuration:")
	print(bestSVM(X, Y, 4))
