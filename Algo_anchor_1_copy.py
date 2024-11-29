import numpy as np
import scipy.sparse as sp
import math
from scipy import linalg
from cvxopt import matrix, solvers


def normalize_data(Y):
    """
    将数据归一化到[0, 1]范围内
    :param data: 要归一化的数据，可以是list或ndarray类型
    :return: 归一化后的数据，与原始数据类型相同
    """
    for i in range(Y.shape[1]):
        x = (x - np.mean(x)) / np.std(x)
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    # debug("Data scaled.")

    return Y


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data


def algo_qp1(X=None, T=None, Y=None, lambda_=None, d=None, numanchor=None,tt=1):
    # m      : the number of anchor. the size of Z is m*n.
    # lambda : the hyper-parameter of regularization term.
    # X      : n*di

    ## initialize
    maxIter = 12
    ap = 1
    m = numanchor
    gama = -1
    i=0
    numclass = len(np.unique(Y))
    numview = len(T)
    numsample = Y.shape[1 - 1]
    W = []  # cell(numview,1)

    A = np.zeros((d, m))

    Z = np.zeros((m, numsample))

    Wt = []
   # h=np.arange(0, numview)# cell(numview,1)
    for i in np.arange(0, numview).reshape(-1):
        if tt==1:
          X[i] = np.transpose(z_score_normalize(X[i]))
       # X[i] = np.transpose(X[i])
        di = X[i].shape[1-1]

        B = np.zeros((di, d))
        W.append(B)
        numsample = Y.shape[1-1]
        m = numanchor
        B1 = np.zeros((numsample, m))
        Wt.append(B1)

    Z[:, 0:m] = np.eye(m)
    alpha1 = np.ones((1, numview)) / (numview)
    o = []
    # opt.disp = 0
    flag = 1
    iter = 0
    ##
    while flag:

        iter = iter + 1
        ## optimize W_i
        AZ = np.dot(A, Z)  # A * Z
        for iv in np.arange(0, numview).reshape(-1):
            C = np.dot(X[iv], np.transpose(AZ))
            print(np.isnan(AZ).any())
            print(np.isnan(X[i]).any())
            print(np.isnan(C).any())
            U, __, V = linalg.svd(C, 0)

            W[iv] = np.dot(U, V)
        for iv in np.arange(0, numview).reshape(-1):
            C = np.dot(T[iv], np.transpose(Z))
            U1, __, V1 = linalg.svd(C, 0)
            Wt[iv] = np.dot(U1, V1)
        ## optimize A
        sumAlpha = 0.0
        sumAlpha1 = 0.0
        part1 = 0
        part2 = 0
        for ia in np.arange(0, numview-1).reshape(-1):
            a = alpha1[0][ia]
            al2 = pow(a, 2)
            sumAlpha = sumAlpha + al2
            part1 = part1 + al2 * np.transpose(W[ia]) @ X[ia] @ np.transpose(Z)
        Unew, __, Vnew = linalg.svd(part1, 0)
        #     A = (part1/sumAlpha) * inv(Z*Z');
        A = Unew @ Vnew
        for ia in np.arange(0, numview).reshape(-1):
            # t=np.transpose(A)@ np.transpose(W[ia])@ X[ia]
            a = alpha1[0][ia]
            al2 = pow(a, 2)
            I = np.eye(m)
            sumAlpha1 = sumAlpha1 + al2 * (I + ap * I)
            # sumAlpha1 = sumAlpha1 + al2*(ap*I)

            part2 = part2 + al2 * (np.transpose(A) @ np.transpose(W[ia]) @ X[ia] + ap * np.transpose(Wt[ia]) @ T[
                ia])  # al2 * np.transpose(W[ia]) @ X[ia] @ np.transpose(Z)
        Z = np.linalg.inv(sumAlpha1) @ part2  # part2/sumAlpha1
        #  H = 6 * sumAlpha * np.eye(m) + 2 * lambda_ * np.eye(m)
        #  H = (H + np.transpose(H)) / 2
        #  t=[]
        #  # [r,q] = chol(H);
        # # options = optimset('Algorithm','interior-point-convex','Display','off')
        #  for ji in np.arange(0,numsample).reshape(-1):
        #      ff = 0
        #      for j in np.arange(0,numview).reshape(-1):
        #          C = W[j] @ A
        #          f = np.transpose(X[j][:,ji])
        #
        #          t=2 * np.transpose(X[j][:,ji])@C
        #
        #          ff = ff - 2 * np.transpose(X[j][:,ji]) @ C - 4 * np.transpose(T[j][:,ji]) @ Wt[j]
        #          if ji == 2947:
        #              print(ff)
        #      G = matrix(-1.0 * np.eye(m))
        #
        #
        #
        #
        #
        #      #G1= np.zeros((m,numsample))
        #      H = matrix(H)
        #      h=matrix(0.0,(m,1))
        #      Q = matrix(1.,(1,m))
        #      b=matrix(1.)
        #      ff=matrix(ff)
        #      if ji==3024:
        #          print(ff)
        #      s = solvers.qp(H, ff, G, h, Q, b)#ff is same
        #      p=s['x']
        #      if ji==21:
        #          print(ff)
        #      if ji==0:
        #          p1=np.array(p).reshape(m,1)
        #      else:
        #          p2=np.array(p).reshape(m,1)
        #          p1 = np.block([p1, p2])
        # t=np.block([t,p1])
        # print(isinstance(H))
        # Z[:,ji] = solvers.qp(P, ff, G, h, Q, b)
        # s = solvers.qp(P, q, A, b1, Aeq, beq)  ## optimize alpha
        M = np.zeros((numview, 1))

        for iv in np.arange(0, numview).reshape(-1):
            M[iv] =ap*pow(np.linalg.norm(T[iv] - Wt[iv] @ Z, 'fro'), 2)
            +pow(np.linalg.norm(X[iv]-W[iv]@ A @ Z, 'fro'),2)
        Mfra = pow(M, -1)
        Q = 1 / sum(Mfra)
        alpha1[0] = (Q * Mfra).reshape(-1)
        # for i in range(1):
        #     alpha1[0][i] = (-((np.linalg.norm(Z[i] - (A[i].T) @ W[i].T @ X[i])) ** 2+ ap * (np.linalg.norm(Z[i] - Wt[i].T @ T[i])) ** 2) / (gama)) ** (1 / (gama - 1))
        #     M = np.zeros((numview + 1, 1))
        term1 = 0
        for iv in np.arange(0, numview).reshape(-1):
            term1 = term1 + alpha1[0][iv] ** 2  # * np.linalg.norm(X[iv] - W[iv] @ A @ Z,'fro') ** 2
            term2 = lambda_ * np.linalg.norm(Z, 'fro') ** 2
            print(iter)
            o.append(term1 + term2)
        if (iter > 9) and (
                    np.abs((o[iter - 2] - o[iter - 1]) / (o[iter - 2])) < 0.001 or iter > maxIter or o[
                iter - 1] < 1e-10):
            #Z = Z*0.01
            #S_hat_tmp = Z.dot(Z.T)
            UU, __, V = linalg.svd(np.transpose(Z), 0)
            flag = 0
    return UU, V, A