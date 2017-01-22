# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:19:07 2017

@author: User
"""
import cv2
import numpy as np
from scipy.io import savemat
import math
import os
import sys
caffe_root = '/home/ke/caffe/caffe-master/'  
sys.path.insert(0, caffe_root + 'python')
import caffe

def align_face(img,detected_points,dsize=256,padding=0):
	'''
	###param description###
	input:
		img:             input face image
		detected_points: key points which are detected in the img
		dsize:           your desire size of resize
		padding:         margin for padding [0,1]

	output:
		aligned face image
	'''
	#standard postions of face points
	mean_face_xy=np.float32([[0.28406391,0.20123456],[0.71155077,0.20123456],[0.49209353,0.42210701],[0.30711025,0.64123456],[0.69541407,0.64123456]])

	#w,h,_ = img.shape
	mean_xy = [[(padding+mean_face_xy[i][0])/(2*padding+1)*dsize,(padding+mean_face_xy[i][1])/(2*padding+1)*dsize] for i in range(len(mean_face_xy))]
	dect_xy = [[detected_points[j*2],detected_points[j*2+1],1] for j in range(len(detected_points)/2)]

	mean_xy_mat = np.matrix(mean_xy)
	dect_xy_mat = np.matrix(dect_xy,dtype=np.float32)
	#affine_m = cv2.getAffineTransform(dect_xy_mat[:3,:],mean_xy_mat[:3,:])
	affine_m = (dect_xy_mat.T*dect_xy_mat).I*dect_xy_mat.T*mean_xy_mat
	affine_img = cv2.warpAffine(img,affine_m.T,(dsize,dsize))
	return affine_img

def get_fea(net,img,layer,key_points,dw=128,dh=128):
	aligned_face=align_face(img,key_points[5:],dsize=128,padding=0.12)
	im_data = cv2.cvtColor(aligned_face,cv2.COLOR_RGB2GRAY)
	net.blobs['data'].reshape(1,1,dw,dh)
	net.blobs['data'].data[...]=im_data
	net.forward()
	# for layer_name, blob in net.blobs.iteritems():
	# 	print layer_name + '\t' + str(blob.data.shape)

	# for layer_name, param in net.params.iteritems():
	# 	print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
	fea=net.blobs[layer].data
	return fea

def get_match_pairs_fea(net,img_path,pairs_tmp,points,point_lst,dw,dh):
	pairs_path=os.path.join(img_path,pairs_tmp[0])
	dir_out=os.listdir(pairs_path)
	pair1=os.path.join(pairs_path,dir_out[int(pairs_tmp[1])-1])
	pair2=os.path.join(pairs_path,dir_out[int(pairs_tmp[2])-1])

	pair1_id_img=os.path.join(pairs_tmp[0],dir_out[int(pairs_tmp[1])-1])
	pair2_id_img=os.path.join(pairs_tmp[0],dir_out[int(pairs_tmp[2])-1])
	idx1=point_lst.index(pair1_id_img)
	idx2=point_lst.index(pair2_id_img)
	print('----------------------------------------------------')
	print ("Processing pairs in "+pairs_path)
	img1=cv2.imread(pair1)
	key_points1=points[idx1].split()
	img2=cv2.imread(pair2)
	key_points2=points[idx2].split()
	print ("extract pair1: ",pair1)	
	fea1=get_fea(net,img1,'eltwise_fc1',key_points1).copy()

	# print("fea1[0,0:2]---->",fea1[0,0:2])
	# print("type,id of fea1: ",type(fea1),id(fea1))
	# print ("extract pair2: ",pair2)	
	fea2=get_fea(net,img2,'eltwise_fc1',key_points2)
	# print("type,id of fea2: ",type(fea2),id(fea2))
	# print("after ...id of fea1,fea2---->",id(fea1),id(fea2))
	# print("fea2[0,0:2]---->",fea2[0,0:2])
	# print("after ...fea1[0,0:2],fea2[0,0:2]---->",fea1[0,0:2],fea2[0,0:2])
	return fea1,fea2

def get_dismatch_pairs_fea(net,img_path,pairs_tmp,points,point_lst,dw,dh):
	pair1_path=os.path.join(img_path,pairs_tmp[0])
	pair2_path=os.path.join(img_path,pairs_tmp[2])
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print("dismatch pairs:",pair1_path,pair2_path)
	dir1_out=os.listdir(pair1_path)
	dir2_out=os.listdir(pair2_path)

	pair1=os.path.join(pair1_path,dir1_out[int(pairs_tmp[1])-1])
	pair2=os.path.join(pair2_path,dir2_out[int(pairs_tmp[3])-1])

	pair1_id_img=os.path.join(pairs_tmp[0],dir1_out[int(pairs_tmp[1])-1])
	pair2_id_img=os.path.join(pairs_tmp[2],dir2_out[int(pairs_tmp[3])-1])

	idx1=point_lst.index(pair1_id_img)
	idx2=point_lst.index(pair2_id_img)

	img1=cv2.imread(pair1)
	key_points1=points[idx1].split()
	img2=cv2.imread(pair2)
	key_points2=points[idx2].split()

	fea1=get_fea(net,img1,'eltwise_fc1',key_points1).copy()
	fea2=get_fea(net,img2,'eltwise_fc1',key_points2)
	return fea1,fea2

def main():
	pairs_list='./extract_fea/pairs.txt'
	points_list='./extract_fea/LFW_points.txt'
	img_path='./extract_fea/LFW/'
	dw,dh=128,128
	caffe.set_mode_cpu()

	model_def = './proto/LightenedCNN_C_deploy.prototxt'
	model_weights = './model/LightenedCNN_C.caffemodel'

	net = caffe.Net(model_def,model_weights,caffe.TEST)  

	with open(pairs_list) as f:
		pairs=f.read()
	pairs=pairs.splitlines()[1:]

	with open(points_list) as f:
		points=f.read()
	points=points.splitlines()
	point_lst=[points[i].split()[0] for i in range(len(points))]

	match_fea=np.zeros((3000,512))
	dismatch_fea=np.zeros((3000,512))
	m_idx=0
	dism_idx=0
	for i in range(len(pairs)):
		print("Total pairs:%d, current is:%d" % (len(pairs),i+1))
		pairs_tmp=pairs[i].split()
		if len(pairs_tmp)==3:
			print("Processing match pairs...")
			fea1,fea2=get_match_pairs_fea(net,img_path,pairs_tmp,points,point_lst,dw,dh)
			fea=np.append(fea1,fea2)
			match_fea[m_idx]=fea
			m_idx=m_idx+1
		elif len(pairs_tmp)==4:
			print("Processing dismatch pairs...")
			fea1,fea2=get_dismatch_pairs_fea(net,img_path,pairs_tmp,points,point_lst,dw,dh)
			fea=np.append(fea1,fea2)
			dismatch_fea[dism_idx]=fea
			dism_idx=dism_idx+1
		else:
			pass

	savemat('lfw_modelc.mat',{'fea_m':match_fea,'fea_dism':dismatch_fea})

if __name__ == '__main__':
	main()

