# coding:utf-8
'''
author : ke.liu@xiaoi
date   : 2017-02-06
'''
import os
import cv2
import caffe
import numpy as np
import sys
import cPickle as pickle


caffe_root = '/home/ke/caffe/caffe-center-loss/caffe-master/'  
model_path = './model/'  
sys.path.insert(0, caffe_root + 'python')

# caffe.set_device(0)
# caffe.set_mode_gpu()

caffe.set_mode_cpu()

net = caffe.Net(model_path + 'resnet50_face_deploy.prototxt',
                model_path + 'resnet_50_face_iter_900000.caffemodel',
                caffe.TEST)

#for layer_name,blob in net.blobs.iteritems():
#    print layer_name + '\t' + str(blob.data.shape)
mean_face=np.load(model_path + 'db_mean_face.npy')
mean_face=mean_face.mean(1).mean(1)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))   # height*width*channel -> channel*height*width
transformer.set_mean('data', mean_face) # mean pixel
transformer.set_raw_scale('data',255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_input_scale('data',0.00390625)
net.blobs['data'].reshape(1,3,224,224)

layer='pool2' #the layer which we need extract feature from

f1 = open('match_pair_resnet50.txt','w')
f2 = open('dismatch_pair_resnet50.txt','w')
count=0
with open('pairs.txt', 'r') as f:
    for line in f.readlines():
        count+=1
        print "processing pair: ",count
        pair = line.strip().split()

        if len(pair) == 3:
            print pair
            name1 = "lfw-deepfunneled/{}/{}_{}.jpg".format(pair[0], pair[0], pair[1].zfill(4))
            name2 = "lfw-deepfunneled/{}/{}_{}.jpg".format(pair[0], pair[0], pair[2].zfill(4))

            #net.blobs['data'].data[0] = transformer.preprocess('data', caffe.io.load_image(name1,color=False))
            net.blobs['data'].data[0] = transformer.preprocess('data', caffe.io.load_image(name1))
            out = net.forward()                  
            x1 = net.blobs[layer].data[0].copy()
            print "shape of x1: ",x1.shape

            net.blobs['data'].data[0] = transformer.preprocess('data', caffe.io.load_image(name2))
            out = net.forward()   
            x2 = net.blobs[layer].data[0].copy()

            x = np.append(x1,x2)
            print "shape of x: ",x.shape
            for i in range(len(x)):
                f1.write('%.3f ' % x[i])
            f1.write('\n')

        if len(pair) == 4:
            print pair
            name1 = "lfw-deepfunneled/{}/{}_{}.jpg".format(pair[0], pair[0], pair[1].zfill(4))
            name2 = "lfw-deepfunneled/{}/{}_{}.jpg".format(pair[2], pair[2], pair[3].zfill(4))

            net.blobs['data'].data[0] = transformer.preprocess('data', caffe.io.load_image(name1))
            out = net.forward()                  
            x1 = net.blobs[layer].data[0].copy()

            net.blobs['data'].data[0] = transformer.preprocess('data', caffe.io.load_image(name2))
            out = net.forward()   
            x2 = net.blobs[layer].data[0].copy()

            x = np.append(x1,x2)
            print "shape of x: ",x.shape
            for i in range(len(x)):
                f2.write('%.3f ' % x[i])
            f2.write('\n')
for layer_name,blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
f1.close()
f2.close()

'''
f1 = open('Img_label2.txt','r')
f2 = open('Img_feature_RM.txt','w')

num = -1
for line in f1.xreadlines():
        num += 1
        name = line.strip().split()[0]
        print num,name

        net.blobs['data'].data[0] = transformer.preprocess('data', caffe.io.load_image(name,color=False))
        out = net.forward()

        v = net.blobs['eltwise_fc1']
        x = np.matrix(v.data[0])

        for i in range(x.shape[1]):
            f2.write('%.3f ' % x[0,i])
        f2.write('\n')

f1.close()
f2.close()
'''
