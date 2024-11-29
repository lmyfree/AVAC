import scipy.io as sio
from time import *
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from Algo_anchor_1_copy import algo_qp1
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

	# Load data
	dataset = 'DBLP4057_GAT_with_idx'
	data = sio.loadmat('{}.mat'.format(dataset))
	if(dataset == 'large_cora'):
		X=data['X']	
		A = data['G']			
		gnd = data['labels']
		gnd = gnd[0, :]
	else:
		X = data['features']
		A = data['net_APA']
		B = data['net_APCPA']
		C = data['net_APTPA']
		#D = data['PTP']
		av=[]
		av.append(A)
		av.append(B)
		av.append(C)
		#av.append(D)
		gnd = data['label']
		gnd = gnd.T
		gnd=np.argmax(gnd, axis=0)
		#gnd = gnd - 1
		#gnd = gnd[0, :]

	p=2
	# Store some variables
	nada=[0.33,0.33,0.33]
	gama=-1
	G=[]
	A_=[]
	X_bar=[]


	N = X.shape[0]
	k = len(np.unique(gnd))
	I = np.eye(N)
	I2 = np.eye(X.shape[1])
	#av = av[1]
	if sp.issparse(X):
		X = X.todense()

	for i in range(3):
	# Normalize A
		A=av[i]
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
	#	ANEW = A + A2
		# An=linalg.cholesky(ANEW)
	#	Pca = PCA(n_components=15*k)
	#	U = Pca.fit_transform(ANEW)

	# Set f(A)
	#	A_.append(np.transpose(U))



		# Set f(A)
		A_.append(A+A2)

		# Set the order of filter
		#G_ = G
		for it in range(p):

	 		X = G[i].dot(X)

		X_bar.append(G[i].dot(X))



	kk = 1

	acc_list = []
	nmi_list = []
	f1_list = []
	ari_list = []
	nowa = []
	nowk = []
	best_acc = []
	best_nmi = []
	best_f1 = []
	best_ari = []
	best_a = []
	best_k = []

	# Set the list of alpha
	#list_a = [1e-1,1,10,100,1000]
	list_a =[100]#,200,210,220]#[0.1,1,10,100,1000]# [180,200,210,220]

	print("f(A)=A+A2")

	# Set the range of filter order k 
	while(kk <= 1):

		#compute
		# for i in range(1):
		# 	X_bar[i] = G[i].dot(X_bar[i])


		#XXt_bar = X_bar.T.dot(X_bar)
		tmp_acc = []
		tmp_nmi = []
		tmp_ari = []
		tmp_f1 = []
		tmp_a = []

		for a in list_a:

			#tmp = np.linalg.inv(I2 + XXt_bar/a)
	        #tmp = X_bar.dot(tmp).dot((X_bar.T))
	        #tmp = I/a -tmp/(a*a)
			#begin_time = time()
			# for i in range(20):
			# 	XtX_bar = 0
			# 	Fasum = 0
			# 	Isum=0
			# 	for j in range(3):
			# 		XtX_bar = XtX_bar + nada[j] * X_bar[j].dot(X_bar[j].T)
			# 	for j in range(3):
			# 		Fasum = Fasum + nada[j] * A_[j]
			# 	for j in range(3):
			# 		Isum = Isum+ nada[j]
			# 	tmp=np.linalg.inv(Isum*a*I+XtX_bar)
			# 	S = tmp.dot(a * Fasum + XtX_bar)
			# 	for j in range(3):
			# 		nada[j]=(-((np.linalg.norm(X_bar[j].T-(X_bar[j].T).dot(S)))**2+a*(np.linalg.norm(S-A_[j]))**2)/gama)**(1/(gama-1))
			# 		# print("nada值")
			# 		# print(nada[j])
			# 	# print("mubiaohanshuzhi")
			# 	# res=0
			# 	# for j in range(3):
			# 	# 	res=res+nada[j]*((np.linalg.norm(X_bar[j].T-(X_bar[j].T).dot(S)))**2+a*(np.linalg.norm(S-A_[j]))**2)+(nada[j])**(gama)
			# 	# print(res)
			# C = 0.5 * (np.fabs(S) + np.fabs(S.T))
			# print("a={}".format(a), "k={}".format(kk), "gamma={}".format(gama))
            #
			# u, s, v = sp.linalg.svds(C, k=k, which='LM')
			
			u, v, _ = algo_qp1(X_bar, A_, gnd, 1, k, k)
			# pdf = PdfPages('dongdblp-out.pdf')
			# cls = np.unique(gnd)
			# X_embedded = TSNE(n_components=2, random_state=33).fit_transform(u)
			# fea_num = [X_embedded[gnd == i] for i in cls]
			# for i, f in enumerate(fea_num):
			#  	if cls[i] in range(10):
			#  		plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='.')
			#  	else:
			#  		plt.scatter(f[:, 0], f[:, 1], label=cls[i])
			# X_embedded.shape
			# # # plt.tight_layout()
			# plt.axis('off')
			# pdf.savefig()
			# # plt.show()
			# pdf.close()

			#row_norm = np.linalg.norm(u, ord=2, axis=1, keepdims=True)
			#u=u/row_norm
			#u[np.isnan(u)] = 0
			kmeans = KMeans(n_clusters=k, random_state=23).fit(u)
			predict_labels = kmeans.predict(u)
			
			# 几个metric

			cm = clustering_metrics(gnd, predict_labels)
			ac, nm, f1, ari = cm.evaluationClusterModelFromLabel(-1, kk, a)
			
			
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

			tmp_acc.append(ac)
			tmp_nmi.append(nm)
			tmp_f1.append(f1)
			tmp_ari.append(ari)
			tmp_a.append(a)

	        
	#         a = a + 50
		nxia = np.argmax(tmp_acc)
		best_acc.append(tmp_acc[nxia])
		best_nmi.append(tmp_nmi[nxia])
		best_f1.append(tmp_f1[nxia])
		best_ari.append(tmp_ari[nxia])
		best_a.append(tmp_a[nxia])
		best_k.append(kk)
		kk += 1
		#G_ = G_.dot(G)
	
	# all of the results
	for i in range(np.shape(acc_list)[0]):
		print("a = {:>.6f}".format(nowa[i]),
	          "k={:>.6f}".format(nowk[i]),
	          "ac = {:>.6f}".format(acc_list[i]),
	          "nmi = {:>.6f}".format(nmi_list[i]),
			  "ari={:>.6f}".format(ari_list[i]),
	          "f1 = {:>.6f}".format(f1_list[i]))
	# the best results for each k
	for i in range(np.shape(best_acc)[0]):
		print("for k={:>.6f}".format(best_k[i]),
	            "the best a = {:>.6f}".format(best_a[i]),
	          "ac = {:>.6f}".format(best_acc[i]),
	          "nmi = {:>.6f}".format(best_nmi[i]),
			  "ari = {:>.6f}".format(best_ari[i]),
	          "f1 = {:>.6f}".format(best_f1[i]))
	    
	# the best result of all experiment
	xia = np.argmax(acc_list)
	print("the best state:")
	print("a = {:>.6f}".format(nowa[xia]),
	          "k={:>.6f}".format(nowk[xia]),
	          "ac = {:>.6f}".format(acc_list[xia]),
	          "nmi = {:>.6f}".format(nmi_list[xia]),
		      "ari = {:>.6f}".format(ari_list[xia]),
	          "f1 = {:>.6f}".format(f1_list[xia]))