import scipy.io as sio
import time
import time
from time import *
# import tensorflow as tf
from utils1 import clustering_accuracy, clustering_f1_score, preprocess_dataset, datagen
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from Algo_anchor_1_copy import algo_qp1
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from itertools import product
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Load data
    dataset = 'wiki'
   # data = sio.loadmat('{}.mat'.format(dataset))

    print('-----------------', dataset, '-----------------')
    As, Xs, labels = datagen(dataset)
    k = len(np.unique(labels))
    views = list(product(As, Xs))

    for v in range(len(views)):
        A, X = views[v]
        tf_idf = dataset in ['acm', 'dblp', 'imdb', 'photos']
        norm_adj, features = preprocess_dataset(A, X, tf_idf=tf_idf, beta=1)

        if type(features) != np.ndarray:
            features = features.toarray()

        if type(norm_adj) == np.matrix:
            norm_adj = np.asarray(norm_adj)

        views[v] = (norm_adj, features)
    for i in range(4):
    # Normalize A
        adj, X = views[i]
#        adj= adj.toarray()
        views[i] = (adj, X)



    # Store some variables
    nada=[1,1,1,1]
    gamas=[-1]
    G=[]
    A_=[]
    X_bar=[]

    p=2
    N = X.shape[0]
    k = len(np.unique(labels))
    I = np.eye(N)
   # I2 = np.eye(X.shape[1])
    if sp.issparse(X):
        X = X.todense()
    gnd = labels
    for i in range(4):
    # Normalize A
        A, X = views[i]
       # A= adj.toarray()
        A = A + I
        D = np.sum(A,axis=1)
        D = np.power(D,-0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        A = D.dot(A).dot(D)

        # Get filter G
        Ls = I - A
        G.append(I - 0.5*Ls)

        # Get the Polynomials of A
        A2 = A.dot(A)

        ANEW = A + A2
        # An=linalg.cholesky(ANEW)
       # Pca = PCA(n_components=k)
       # U = Pca.fit_transform(ANEW)

    # Set f(A)
       # A_.append(np.transpose(ANEW))

        A_.append(A+A2)

        # Set the order of filter
        #G_ = G
        for it in range(p):
            X = G[i].dot(X)
        X_bar.append(G[i].dot(X))
    kk = 4

    acc_list = []
    nmi_list = []
    f1_list = []
    ari_list = []
    nowa = []
    nowk = []
    nowgama=[]
    best_acc = []
    best_nmi = []
    best_f1 = []
    best_ari = []
    best_a = []
    best_k = []
    best_gama=[]

    # Set the list of alpha
    list_a = [15]
    #list_a=[1,2,5,10]
    #cons_a=0.01
    print("f(A)=A+A2")

    # Set the range of filter order k
    while(kk <= 4):

        #compute
        for i in range(4):
            X_bar[i] = G[i].dot(X_bar[i])


        #XXt_bar = X_bar.T.dot(X_bar)
        tmp_acc = []
        tmp_nmi = []
        tmp_f1 = []
        tmp_ari = []
        tmp_a = []
        tmp_gama=[]
        alpha=2
        # final=[]
        for a in list_a:

            #tmp = np.linalg.inv(I2 + XXt_bar/a)
            #tmp = X_bar.dot(tmp).dot((X_bar.T))
            #tmp = I/a -tmp/(a*a)
            # for gama in gamas:
            #     for i in range(20):
            #         XtX_bar = 0
            #         Fasum = 0
            #         Isum=0
            #         for j in range(4):
            #             XtX_bar = XtX_bar + nada[j] * X_bar[j].dot(X_bar[j].T)
            #         for j in range(4):
            #             Fasum = Fasum + nada[j] * A_[j]
            #         for j in range(4):
            #             Isum = Isum+ nada[j]
            #         tmp=np.linalg.inv(Isum*a*I+XtX_bar)
            #         S = tmp.dot(a * Fasum + XtX_bar)
            #         for j in range(4):
            #             nada[j]=(-((np.linalg.norm(X_bar[j].T-(X_bar[j].T).dot(S)))**2+a*(np.linalg.norm(S-A_[j]))**2)/(gama))**(1/(gama-1))
                        # print("nada值")
                        # print(nada[j])
                    # print("mubiaohanshuzhi")
                    # res = 0
                    # for j in range(2):
                    #     res = res + nada[j] * ((np.linalg.norm(X_bar[j].T - (X_bar[j].T).dot(S))) ** 2 + a * (
                    #     np.linalg.norm(S - A_[j])) ** 2) + (nada[j]) ** (gama)
                    # final.append(res)
                    # print(res)
                # term1=0
                # term2=0
                # for i in range(2):
                #     term1=term1+nada[i]*((np.linalg.norm(X_bar[j].T-(X_bar[j].T).dot(S)))**2+a*(np.linalg.norm(S-A_[j])))
                # for i in range(2):
                #     term2=term2+cons_a*(nada[i])**gama

               
                #print("第一项的值%e"%term1)
                # sio.savemat("a.mat",{'res':final})
                #print("第一项的值%e" % term2)
                # C = 0.5 * (np.fabs(S) + np.fabs(S.T))
                # print("a={}".format(a), "k={}".format(kk),"gamma={}".format(gama))
                #
                # #u, s, v = sp.linalg.svds(C, k=k, which='LM')
                #
                # u, s, v = sp.linalg.svds(C, k=k, which='LM')
                u, v, _ = algo_qp1(X_bar, A_, gnd, 1, k, k)
                row_norm = np.linalg.norm(u, ord=2, axis=1, keepdims=True)
                u = u / row_norm
                kmeans = KMeans(n_clusters=k, random_state=23).fit(u)
                predict_labels = kmeans.predict(u)
                # pdf = PdfPages('dongwiki-out.pdf')
                # cls = np.unique(gnd)
                # X_embedded = TSNE(n_components=2, random_state=33).fit_transform(u)
                # fea_num = [X_embedded[gnd == i] for i in cls]
                # for i, f in enumerate(fea_num):
                #     if cls[i] in range(10):
                #         plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='.')
                #     else:
                #         plt.scatter(f[:, 0], f[:, 1], label=cls[i])
                # X_embedded.shape
                # # plt.tight_layout()
                # plt.axis('off')
                # pdf.savefig()
                # plt.show()
                # pdf.close()



                # 几个metric

                cm = clustering_metrics(gnd, predict_labels)
                
                ac, nm, f1,ari = cm.evaluationClusterModelFromLabel(-1,kk,a)

                print(
                    'acc_mean: {}'.format(ac),
                    'nmi_mean: {}'.format(nm),
                    'f1_mean: {}'.format(f1),
                    'ari_mean: {}'.format(ari),
                    'max_element :{}'.format(np.max(A_)),
                    '\n' * 2)
                acc_list.append(ac)
                nmi_list.append(nm)
                f1_list.append(f1)
                ari_list.append(ari)
                nowa.append(a)
                nowk.append(kk)
                nowgama.append(-1)

                tmp_acc.append(ac)
                tmp_nmi.append(nm)
                tmp_f1.append(f1)
                tmp_ari.append(ari)
                tmp_a.append(a)
                tmp_gama.append(-1)


    #         a = a + 50
        nxia = np.argmax(tmp_acc)
        best_acc.append(tmp_acc[nxia])
        best_nmi.append(tmp_nmi[nxia])
        best_f1.append(tmp_f1[nxia])
        best_ari.append(tmp_ari[nxia])
        best_a.append(tmp_a[nxia])
        best_gama.append(tmp_gama[nxia])
        best_k.append(kk)
        kk += 1
        #G_ = G_.dot(G)

    # all of the results
    for i in range(np.shape(acc_list)[0]):
        print("a = {:>.6f}".format(nowa[i]),
              "k={:>.6f}".format(nowk[i]),
              "gama={:>.6f}".format(nowgama[i]),
              "ac = {:>.6f}".format(acc_list[i]),
              "nmi = {:>.6f}".format(nmi_list[i]),
              "ari={:>.6f}".format(ari_list[i]),
              "f1 = {:>.6f}".format(f1_list[i]))
    # the best results for each k
    for i in range(np.shape(best_acc)[0]):
        print("for k={:>.6f}".format(best_k[i]),
                "the best a = {:>.6f}".format(best_a[i]),
              "gama={:>.6f}".format(best_gama[i]),
              "ac = {:>.6f}".format(best_acc[i]),
              "nmi = {:>.6f}".format(best_nmi[i]),
              "ari = {:>.6f}".format(best_ari[i]),
              "f1 = {:>.6f}".format(best_f1[i]))

    # the best result of all experiment
    xia = np.argmax(acc_list)
    print("the best state:")
    print("a = {:>.6f}".format(nowa[xia]),
              "k={:>.6f}".format(nowk[xia]),
               "gama={:>.6f}".format(nowgama[xia]),
              "ac = {:>.6f}".format(acc_list[xia]),
              "nmi = {:>.6f}".format(nmi_list[xia]),
              "ari = {:>.6f}".format(ari_list[xia]),
              "f1 = {:>.6f}".format(f1_list[xia]))