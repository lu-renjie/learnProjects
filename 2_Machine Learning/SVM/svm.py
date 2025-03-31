import numpy as np
from functools import partial


def rbf(X, Y, gamma):
    X = X[:, None, :]
    Y = Y[None, :, :]
    temp = np.power(X - Y, 2).sum(axis=2)
    return np.exp(- temp * gamma)


class SVM:
    def __init__(self, kernel='rbf', C=1, sigma=1, tol=1e-3):
        self.m = 0
        self.X = None
        self.Y = None

        self.C = C
        self.K = None
        self.alpha = None
        self.b = 0
        self.E = None

        self.tol = tol

        assert tol <= 0.1
        assert kernel in ('linear', 'rbf')

        if kernel == 'linear':
            self.kernel = lambda X, Y: np.dot(X, Y.T)
        else:
            self.kernel = partial(rbf, gamma=1 / sigma**2)

    def fit(self, X, Y):
        self.m = len(X)
        self.X = X
        self.Y = Y
        self.E = self.b - self.Y  # 由于alpha初始化为0，因此y_if(x_i)也为0，E_i = b - Y_i
        self.K = self.kernel(X, X)
        self.alpha = np.zeros(self.m)

        # self.__simpleSMO()
        self.__PlattSMO()

        nonzero = np.nonzero(self.alpha)[0]
        self.m = len(nonzero)
        self.alpha = self.alpha[nonzero]
        self.X = self.X[nonzero, :]
        self.Y = self.Y[nonzero]
        self.K = self.K[nonzero, nonzero]

    def predict(self, X):
        temp = self.alpha[:, None] * self.Y[:, None] * self.kernel(self.X, X).reshape(self.m, -1)
        temp = self.b + temp.sum(axis=0)
        return temp

    def __simpleSMO(self):
        it = 0
        max_iter = self.m * self.m
        while True:
            meet_kkt_num = 0
            for s1 in range(self.m):
                if self.__meet_KKT(s1):
                    meet_kkt_num += 1
                    continue
                for s2 in range(self.m):
                    if s1 == s2:
                        continue
                    self.__step(s1, s2)
                    it += 1
                    if it % max_iter == 0:
                        print('Warning: 到达最大迭代次数, 可能未收敛')
                        return
            if meet_kkt_num == self.m:
                break
        print('迭代次数:', it)

    def __PlattSMO(self):
        """
        基于Platt的启发式搜索做了一些修改，不能保证收敛到最优解，但是速度比较快
        """
        losses, it = [], 0
        diag = np.diag(self.K)
        quad = np.power(diag, 2)
        coef = (quad + quad.reshape(self.m, 1) - 2 * diag * self.K) * self.Y
        coef[coef == 0] += self.tol

        def selectJ(i):
            # 这里alpha > 0说明还可以减少，因此求负数里的最大值
            diff = (self.E[i] - self.E) / coef[i, :]
            j = np.argmax(diff * (self.alpha < self.C) - diff * (self.alpha > 0))
            while j == i:
                j = np.random.randint(0, self.m)
            return j

        while it < 100:  # 最多迭代100次
            not_meet_kkt = 0
            change = False

            for s1 in range(self.m):  # 遍历不满足KKT条件的变量
                if self.__meet_KKT(s1):
                    continue
                not_meet_kkt += 1
                s2 = selectJ(s1)
                if self.__step(s1, s2):
                    change = True
            it += 1

            if not_meet_kkt == 0:  # 一般来说，容差很大之后才满足KKT条件
                print('容差:', self.tol)
                break

            if not change:  # 迭代不再更新了，增大容差
                self.tol *= 5
                change = True

            while change:  # 对于线性可分的数据这里的选择比较重要，因为支持向量比较稀疏
                change = False
                unbound = np.argsort((self.E != 0) * (self.alpha > 0) * (self.alpha < self.C))
                for s1 in unbound[::-1]:
                    s2 = selectJ(s1)
                    if self.__step(s1, s2):
                        change = True
                it += 1
        print('迭代次数:', it)

    def __step(self, s1, s2):
        E1, E2 = self.E[s1], self.E[s2]
        a2 = self.alpha[s2] + self.Y[s2] * (E1 - E2) / (self.K[s1, s1] + self.K[s2, s2] - 2 * self.K[s1, s2])
        a2 = self.__clip(a2, s1, s2)
        a1 = self.alpha[s1] + self.Y[s1] * self.Y[s2] * (self.alpha[s2] - a2)

        temp1 = (a1 - self.alpha[s1]) * self.Y[s1]
        temp2 = (a2 - self.alpha[s2]) * self.Y[s2]
        b1 = self.b - E1 - temp1 * self.K[s1, s1] - temp2 * self.K[s1, s2]
        b2 = self.b - E2 - temp1 * self.K[s2, s1] - temp2 * self.K[s2, s2]

        change1 = a1 - self.alpha[s1]
        change2 = a2 - self.alpha[s2]
        b_old = self.b

        if abs(change1) + abs(change2) < self.tol:
            return False

        self.alpha[s1] = a1
        self.alpha[s2] = a2
        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        self.E = self.E + change1 * self.Y[s1] * self.K[s1, :] + change2 * self.Y[s2] * self.K[s2, :] + (self.b - b_old)
        return True

    def __meet_KKT(self, i):
        temp = self.Y[i] * self.E[i]
        if 0 < self.alpha[i] < self.C and abs(temp) > self.tol:
            return False
        elif self.alpha[i] == 0 and temp < -self.tol:
            return False
        elif self.alpha[i] == self.C and temp > self.tol:
            return False
        return True

    def __clip(self, alpha, s1, s2):
        alpha1_old, alpha2_old = self.alpha[s1], self.alpha[s2]
        y1, y2 = self.Y[s1], self.Y[s2]

        if y1 * y2 == 1:
            L = max(0, alpha1_old + alpha2_old - self.C)
            U = min(self.C, alpha1_old + alpha2_old)
        else:
            L = max(0, alpha2_old - alpha1_old)
            U = min(self.C, self.C + alpha2_old - alpha1_old)

        if alpha > U:
            return U
        elif alpha < L:
            return L
        else:
            return alpha
    
    def __loss(self):
        temp = self.K * (self.alpha * self.alpha.reshape(self.m, 1)) * (self.Y * self.Y.reshape(self.m, 1))
        temp = self.alpha.sum() - 0.5 * temp.sum()
        return temp
