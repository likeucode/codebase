import cv2
import numpy as np
import math

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
	

def main():
	point_list='E:/data/casia/face_point.list'
	imgdata_path='E:/data/casia/'
	with open(point_list) as f:
		detected_points=f.read()

	list_points=detected_points.splitlines()

	one_sample=list_points[0].split()

	print "one_sample:",one_sample
	img=cv2.imread(imgdata_path+one_sample[0])
	affine_img=align_face(img,one_sample[5:],dsize=256,padding=0.12)
	cv2.imwrite("aligned_img.jpg",affine_img)
	#cv2.imshow("AffinedImg",affine_img)

if __name__== "__main__":
	main()
