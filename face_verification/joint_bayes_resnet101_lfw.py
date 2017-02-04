# coding=utf-8
import numpy as np
from scipy.io import loadmat,savemat
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import metrics
import cPickle as pickle

from common import *

def data_normalize(feature):
    
    std = np.std(feature, axis=0)
    mean=np.mean(feature,axis=0)
    feature=(feature-mean)/std

    return feature

def preprocess_data(data_path):
	with open(data_path+'Img_feature_RES.txt') as fea:
		fea_str=fea.read()
	fea_strlines=fea_str.splitlines()

	with open(data_path+'Img_label_RES.txt') as f:
		raw_label=f.read()
	raw_label_strlines=raw_label.splitlines()

	feature=np.zeros((len(fea_strlines),len(fea_strlines[0].split())))
	label=np.zeros(len(raw_label_strlines))

	for i in range(feature.shape[0]):
		feature[i]=np.float32(fea_strlines[i].split())
		label[i]=np.int(raw_label_strlines[i].split()[1])

        feature = preprocessing.scale(feature,axis=0)
	savemat("./resnet_result/feature_vs_label.mat",{"feature":feature,"label":label})

	#return feature,label

def pca_train(data,n_components=128,fold="./resnet_result/"):
    print_info("PCA training (n_components=%d)..." % n_components)
    pca = IncrementalPCA(n_components=n_components)
    pca.fit(data)
    data_to_pkl(pca, fold + "pca_model.pkl")
    print_info("PCA done.")
    return pca

# Before training,the mean must be substract
def joint_bayesian_train(trainingset, label, fold="./resnet_result/", threshold=1e-8, max_iter=100):
    print "Training joint bayes......"
    print "shape of trainingset: ",trainingset.shape
    print "shape of training label: ",label.shape

    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    print "After np.unique the shape of classes,labels: ",classes.shape,labels.shape
    # the total people num
    n_class = len(classes)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image)
    maxNumberInOneClass = 0
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels == i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]

        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print "prepare done, maxNumberInOneClass=", maxNumberInOneClass

    u = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp = 0
    for i in range(n_class):
        # the mean of cur[i]
        u[:, i] = np.mean(cur[i], 0)
        b = u[:, i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp + n_same_label] = cur[i].T - b
            nowp += n_same_label

    Su = np.cov(u.T, rowvar=0)
    Sw = np.cov(ep.T, rowvar=0)
    oldSw = Sw
    SuFG = {}
    SwG = {}
    min_convergence = 1
    ith_model = 0
    for l in range(max_iter):
        F = np.linalg.pinv(Sw)
        u = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
                # G = −(mS μ + S ε )−1*Su*Sw−1
                G = -np.dot(np.dot(np.linalg.pinv(mi * Su + Sw), Su), F)
                # Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F + mi * G))
                # Sw*G for e
                SwG[mi] = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
            # formula 7 in suppl_760
            u[:, i] = np.sum(np.dot(SuFG[nn_class], cur[i].T), 1)
            # formula 8 in suppl_760
            ep[:, nowp:nowp + nn_class] = cur[i].T + np.sum(np.dot(SwG[nn_class], cur[i].T), 1).reshape(n_dim, 1)
            nowp = nowp + nn_class

        Su = np.cov(u.T, rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw - oldSw) / np.linalg.norm(Sw)
        print_info("Iterations-" + str(l) + ": " + str(convergence))
        if convergence < threshold:
            print "Convergence: ", l, convergence
            break;
        oldSw = Sw

        if convergence < min_convergence:
            min_convergence = convergence
            F = np.linalg.pinv(Sw)
            G = -np.dot(np.dot(np.linalg.pinv(2 * Su + Sw), Su), F)
            A = np.linalg.pinv(Su + Sw) - (F + G)
            data_to_pkl(G, fold + "G.pkl")
            data_to_pkl(A, fold + "A.pkl")
            ith_model += 1

    F = np.linalg.pinv(Sw)
    G = -np.dot(np.dot(np.linalg.pinv(2 * Su + Sw), Su), F)
    A = np.linalg.pinv(Su + Sw) - (F + G)
    data_to_pkl(G, fold + "G_con.pkl")
    data_to_pkl(A, fold + "A_con.pkl")

    return A, G

# ratio of similar,the threshold we always choose in (-1,-2)
def Verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(
        np.dot(np.transpose(x1), G), x2)
    return float(ratio)

def get_ratios(A, G, pair_list, data):
    distance = []
    for pair in pair_list:
        ratio = Verify(A, G, data[pair[0]], data[pair[1]])
        distance.append(ratio)

    return distance

def make_pairlist(test_label):
	print "Making pair list for testing......"
	label_lst=test_label.tolist()
	test_class=np.unique(test_label)
	
	intra_pair=[]
	extra_pair=[]
	for i in range(test_class.shape[0]):
		#IntraPersonPair
		if label_lst.count(test_class[i]) >= 2:
			intra_index = np.where(test_label == test_class[i])[0]
			np.random.shuffle(intra_index)
			intra_pair.append([intra_index[0],intra_index[1]])
                        np.random.shuffle(intra_index)
            		intra_pair.append([intra_index[0],intra_index[1]])

		#ExtraPersonPair
		idx = np.where(test_label == test_class[i])[0]
		extra_index = np.where(test_label != test_class[i])[0]
		np.random.shuffle(extra_index)
		extra_pair.append([idx[0],extra_index[0]])
        	np.random.shuffle(extra_index)
        	extra_pair.append([idx[0],extra_index[0]])

	print "lenght of intra pair :",len(intra_pair)
	print "lenght of extra pair :",len(extra_pair)
	savemat("data_pairlist.mat",{'IntraPersonPair':intra_pair,'ExtraPersonPair':extra_pair})

	#return intra_pair,extra_pair

def lfw_pairlist(pair_path):
    with open(pair_path+"match_result.txt") as match:
        m=match.read()
    mlines=m.splitlines()

    with open(pair_path+"dismatch_result.txt") as dismatch:
        dism=dismatch.read()
    dismlines=dism.splitlines()

    fea_m=np.zeros((len(mlines),len(mlines[0].split())))
    label_m=np.ones(len(mlines))
    for i in range(fea_m.shape[0]):
        fea_m[i]=np.float32(mlines[i].split())

    fea_dism=np.zeros((len(dismlines),len(mlines[0].split())))
    label_dism=np.zeros(len(dismlines))
    for i in range(fea_dism.shape[0]):
        fea_dism[i]=np.float32(dismlines[i].split())

    fea_m = preprocessing.scale(fea_m,axis=0)
    fea_dism = preprocessing.scale(fea_dism,axis=0)

    savemat("./resnet_result/lfw_pair.mat",{"fea_m":fea_m,"label_m":label_m,"fea_dism":fea_dism,"label_dism":label_dism})

def lfw_getratios(A,G,data_pair):
    distance=[]
    mid = data_pair.shape[1]/2
    for i in range(data_pair.shape[0]):
        ratio=Verify(A,G,data_pair[i,:mid],data_pair[i,mid:])
        distance.append(ratio)

    return distance
def lfw_pca_test(intra,extra,result_fold='./resnet_result/'):
    print "Executing test in lfw......"

    with open(result_fold + 'A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open(result_fold + 'G.pkl', 'rb') as f:
        G = pickle.load(f)

    dist_Intra = lfw_getratios(A, G, intra)
    dist_Extra = lfw_getratios(A, G, extra)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))
    for i in range(0, len(dist_all), 100):
        print(dist_all[i], label[i])
    data_to_pkl({'distance': dist_all, 'label': label}, result_fold + 'result.pkl')
    savemat(result_fold+'result_pca.mat',{'distance':dist_all,'label':label})

    print "test done"

def lfw_test(lfw_pair_path,result_fold='./resnet_result/'):
    print "Executing test in lfw......"
    data = loadmat(lfw_pair_path)
    test_Intra = data['fea_m']
    test_Extra = data['fea_dism']

    with open(result_fold + 'A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open(result_fold + 'G.pkl', 'rb') as f:
        G = pickle.load(f)

    dist_Intra = lfw_getratios(A, G, test_Intra)
    dist_Extra = lfw_getratios(A, G, test_Extra)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))
    for i in range(0, len(dist_all), 100):
        print(dist_all[i], label[i])
    data_to_pkl({'distance': dist_all, 'label': label}, result_fold + 'result.pkl')
    savemat(result_fold+'result.mat',{'distance':dist_all,'label':label})

    print "test done"    
def excute_test(pairlist, test_data, result_fold='./resnet_result/'):
    print "Executing test......"
    pair_list = loadmat(pairlist)
    test_Intra = pair_list['IntraPersonPair']
    test_Extra = pair_list['ExtraPersonPair']

    with open(result_fold + 'A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open(result_fold + 'G.pkl', 'rb') as f:
        G = pickle.load(f)

    dist_Intra = get_ratios(A, G, test_Intra, test_data)
    dist_Extra = get_ratios(A, G, test_Extra, test_data)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))
    for i in range(0, len(dist_all), 100):
        print(dist_all[i], label[i])
    data_to_pkl({'distance': dist_all, 'label': label}, result_fold + 'result.pkl')
    savemat(result_fold+'result.mat',{'distance':dist_all,'label':label})

    print "test done"

def excute_performance(file_path, t_s=-20, t_e=20, t_step=1):
    with open(file_path, "rb") as f:
        result = pickle.load(f)
        dist = result['distance']
        y = result['label']
        print y
        print "test size: ", y.shape
        print "negative size: ", y[y == 0].shape
        print "postive size: ", y[y == 1].shape

        draw_list = []
        while (t_s < t_e):
            pre = dist >= t_s
            y = (y == 1)
            report = metrics.classification_report(y_true=y, y_pred=pre)
            print "threshold: ", t_s
            print report

            report_result = report_format(report)
            draw_list.append([report_result, t_s])
            t_s += t_step

        save_draw_file(draw_list)

def main():
    data_path='./data/'
    result_path='./resnet_result/'

    #preprocess_data(data_path)

    feature_label=loadmat(result_path+'feature_vs_label.mat')
    feature=feature_label['feature']
    label=feature_label['label']
    label=label.T
    label=label[:,0]

    print "shape of feature in main:",feature.shape
    print "shape of label in label:",label.shape
    '''
    #    Without PCA
    lfw_pairlist(data_path)
    joint_bayesian_train(feature,label,threshold=1e-16, max_iter=500,fold="./resnet_result/")
    lfw_test(result_path+'lfw_pair.mat',result_fold='./resnet_result/')
    '''
    
    #    With PCA
	####train####
    pca_model=pca_train(feature,n_components=512)
    feature = pca_model.transform(feature)
    feature=preprocessing.scale(feature,axis=0)
    joint_bayesian_train(feature,label,threshold=1e-14, max_iter=200,fold="./resnet_result/")

	####test####
        
    data = loadmat(result_path+'lfw_pair.mat')
    intra = data['fea_m']
    extra = data['fea_dism']
    feature_point=intra.shape[1]/2
    intra1 = pca_model.transform(intra[:,:feature_point])
    intra2 = pca_model.transform(intra[:,feature_point:])
    extra1 = pca_model.transform(extra[:,:feature_point])
    extra2 = pca_model.transform(extra[:,feature_point:])
    intra = np.append(intra1,intra2,axis=1)
    extra = np.append(extra1,extra2,axis=1)
    intra = preprocessing.scale(intra,axis=0)
    extra = preprocessing.scale(extra,axis=0)

    lfw_pca_test(intra,extra)
    

    excute_performance(result_path+"result.pkl", -60, 0, 0.2)
if __name__ == '__main__':
	main()
