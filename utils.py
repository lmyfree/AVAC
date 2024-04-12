from scipy import io
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
import pickle
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import os
from ogb.nodeproppred import NodePropPredDataset, Evaluator
from sklearn.neighbors import kneighbors_graph

def acm():
  dataset = "data/ACM"
  data = io.loadmat('{}.mat'.format(dataset))

  X = data['features']
  A = data['PAP']
  B = data['PLP']

  Xs = []
  As = []

  Xs.append(X.toarray())
  As.append(A.toarray())
  As.append(B.toarray())

  labels = data['label']
  labels = labels.reshape(-1)

  return As, Xs, labels

def dblp():
  dataset = "data/DBLP"
  data = io.loadmat('{}.mat'.format(dataset))

  X = data['features']
  A = data['net_APTPA']
  B = data['net_APCPA']
  C = data['net_APA']

  Xs = []
  As = []

  Xs.append(X.toarray())
  As.append(A.toarray())
  As.append(B.toarray())
  As.append(C.toarray())

  labels = data['label']
  labels = labels.reshape(-1)

  return As, Xs, labels


def imdb():
  dataset = "data/IMDB"
  data = io.loadmat('{}.mat'.format(dataset))
  
  X = data['features']
  A = data['MAM']
  B = data['MDM']
  
  Xs = []
  As = []

  Xs.append(X.toarray())
  As.append(A.toarray())
  As.append(B.toarray())

  labels = data['label']
  labels = labels.reshape(-1)

  return As, Xs, labels


def photos():
  dataset = 'Amazon_photos'
  data = io.loadmat('{}.mat'.format(dataset))
  
  X = data['features'].toarray().astype(float)
  A = data.get('adj')
  labels = data['label']
  labels = labels.reshape(-1)
  
  As = [A, A]
  
  Xs = [X, np.log2(1+X)]

  return As, Xs, labels
  
  

def wiki():
  data = io.loadmat(os.path.join('', f'wiki.mat'))
  X = data['fea'].toarray().astype(float)
  A = data.get('W')
  labels = data['gnd'].reshape(-1)
  
  As = [A, kneighbors_graph(X, 5, metric='cosine')]
  Xs = [X, np.log2(1+X)]

  return As, Xs, labels
def BlogCatalog():
    dataset = 'BlogCatalog'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['Attributes'].toarray().astype(float)
    A = data['Network'].toarray()#.get('W')
    #X=np.vstack([X, X])
   # X=np.repeat(X,2, axis=0)
    labels = data['Label']
    labels = labels.reshape(-1)
   # labels = np.repeat(labels, 2, axis=0)
    #labels = np.hstack([labels,labels])
   # A = np.repeat(A, 2, axis=0)
   # A = np.repeat(A, 2, axis=1)
    As = [A, A]

    Xs = [X, X @ X.T]

    return As, Xs, labels
def Flickr():
    dataset = 'Flickr'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['Attributes'].toarray().astype(float)
    A = data['Network'].toarray()#.get('W')
    labels = data['Label']
    labels = labels.reshape(-1)

    As = [A, A]

    Xs = [X, X @ X.T]

    return As, Xs, labels

def arxiv():
    dataset = 'arxiv'
    #data = io.loadmat('{}.mat'.format(dataset))
    dataset = NodePropPredDataset(name=f'ogbn-arxiv')
    data = dataset[0]
    adj = sp.coo_matrix(
        (np.ones(1166243), (data[0]["edge_index"][0], data[0]["edge_index"][1])),
        shape=(169343, 169343))

    #X = data['Attributes'].toarray().astype(float)
    X = data[0]["node_feat"]#.toarray()#.get('W')
    labels = data[1]
    labels = labels.reshape(-1)
    with open('adj.pkl', 'wb') as f:
        pickle.dump(adj, f)
    with open('X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    del data, dataset
   # adj=adj.toarray()
    As = [adj]

    Xs = [X,X]
    del adj, X
    return As, Xs, labels

def citeseer():
    dataset = 'citeseer'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['fea'].astype(float)
    A = data.get('W')
    labels = data['gnd']
    labels = labels.reshape(-1)

    As = [A, A]

    Xs = [X, np.log2(1+X)]

    return As, Xs, labels
def com():
    dataset = 'computers'
    data = np.load('computers.npz')

    X = data['attr_data']
    A = data['adj_data']
    labels = data['labels']
    labels = labels.reshape(-1)

    As = [A, A]

    Xs = [X, X @ X.T]

    return As, Xs, labels

def datagen(dataset):
  if dataset == 'imdb': return imdb()
  if dataset == 'dblp': return dblp()
  if dataset == 'acm': return acm()
  if dataset == 'Amazon_photos': return photos()
  if dataset == 'wiki': return wiki()
  if dataset == 'Flickr': return Flickr()
  if dataset == 'BlogCatalog': return BlogCatalog()
  if dataset == 'arxiv': return arxiv()


def preprocess_dataset(adj, features, tf_idf=False, beta=1):
  adj = adj + beta * sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  adj = r_mat_inv.dot(adj)

  if tf_idf:
      features = TfidfTransformer(norm='l2').fit_transform(features)
  else:
      features = normalize(features, norm='l2')

  return adj, features


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def cmat_to_psuedo_y_true_and_y_pred(cmat):
        y_true = []
        y_pred = []
        for true_class, row in enumerate(cmat):
            for pred_class, elm in enumerate(row):
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
        return y_true, y_pred

def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, **kwargs):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)

