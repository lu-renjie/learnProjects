import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from SVM.svm import SVM


def plot_decision_boundary(model, X, point_num):
    l = X.min(axis=0) * 2
    u = X.max(axis=0) * 2
    x, y = np.meshgrid(np.linspace(l[0], u[0], point_num), np.linspace(l[1], u[1], point_num))
    xy = np.stack([x, y], axis=2).reshape(-1, 2)
    prediction = model.predict(xy).reshape(point_num, point_num)
    plt.contourf(x, y, prediction, levels=[-1, 0, 1], alpha=0.5, cmap='Reds')


def main(sample_num, linear):
    if linear:
        X, Y = datasets.make_blobs(n_samples=sample_num, n_features=2, centers=2, cluster_std=1)
        svm = SVM('linear', tol=1e-3)
    else:
        X, Y = datasets.make_circles(n_samples=sample_num, noise=0.15, factor=0.2)
        svm = SVM('rbf', sigma=2, tol=1e-3)

    X -= X.mean(axis=0)
    Y[Y == 0] = -1

    t = time.time()
    svm.fit(X, Y)
    print('用时:', time.time() - t)
    print('准确率:', ((svm.predict(X) * Y) > 0).sum() / sample_num)

    plot_decision_boundary(svm, X, point_num=200)
    plt.scatter(X[:, 0], X[:, 1], c=Y / 2 + 1, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main(1000, linear=True)
