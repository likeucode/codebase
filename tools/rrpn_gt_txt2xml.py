#############################
#author : ke.liu@xiaoi
#date   : 04/11/2017
#usage  : convert text to xml for icdar2013 dataset
#############################
import cv2
import numpy as np
import os
from xml.dom.minidom import Document

def txt2xml(gt_src,gt_filename,gt_dst,sw,sh):
    doc = Document()  
    annotation = doc.createElement('annotation') 
    doc.appendChild(annotation)
    
    folder = doc.createElement('folder') 
    folder.appendChild(doc.createTextNode('HUST_TR400')) ##train
    #folder.appendChild(doc.createTextNode('Challenge2_Test_Task12_Images')) ##test
    filename = doc.createElement('filename') 
    filename.appendChild(doc.createTextNode(filter(str.isdigit,gt_filename)+'.jpg'))
    #filename.appendChild(doc.createTextNode('img_'+filter(str.isdigit,gt_filename)+'.jpg'))##test
    size=doc.createElement('size')

    width=doc.createElement('width')
    width.appendChild(doc.createTextNode(sw))
    height=doc.createElement('height')
    height.appendChild(doc.createTextNode(sh))

    size.appendChild(width)
    size.appendChild(height)

    annotation.appendChild(folder)
    annotation.appendChild(filename)
    annotation.appendChild(size)

    with open(gt_src+gt_filename) as f:
        text=f.read()
        labelLines=text.splitlines()
    for i in range(len(labelLines)):
        label=labelLines[i].split(' ') 
        
        xctr=np.int32(label[2])+np.int32(label[4])/2
        yctr=np.int32(label[3])+np.int32(label[5])/2
        
        objectNode=doc.createElement('object')

        nameNode=doc.createElement('name')
        #nameNode.appendChild(doc.createTextNode(label[-1][1:-1]))##train
        #nameNode.appendChild(doc.createTextNode(label[-1][2:-1]))##test
        nameNode.appendChild(doc.createTextNode('text'))

        difficultNode=doc.createElement('difficult')
        difficultNode.appendChild(doc.createTextNode(label[1]))

        bdnboxNode=doc.createElement('bndbox')
        xctrNode=doc.createElement('x_ctr')      
        xctrNode.appendChild(doc.createTextNode(str(xctr)))
        yctrNode=doc.createElement('y_ctr')
        yctrNode.appendChild(doc.createTextNode(str(yctr)))
        wNode=doc.createElement('w')
        wNode.appendChild(doc.createTextNode(label[4]))
        hNode=doc.createElement('h')
        hNode.appendChild(doc.createTextNode(label[5]))	

        angleNode=doc.createElement('angle')
        angle=float(label[6])/np.pi*180.0
        print "angle: ",angle
        angleNode.appendChild(doc.createTextNode(str(angle)))

        bdnboxNode.appendChild(xctrNode)
        bdnboxNode.appendChild(yctrNode)
        bdnboxNode.appendChild(wNode)
        bdnboxNode.appendChild(hNode);bdnboxNode.appendChild(angleNode)

        objectNode.appendChild(nameNode)
        objectNode.appendChild(difficultNode)
        objectNode.appendChild(bdnboxNode)

        annotation.appendChild(objectNode)

    fp = open(gt_dst+filter(str.isdigit,gt_filename)+'.xml', 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()

def main(gt_src,gt_dst,img_db):
	
    gt_filenames=os.listdir(gt_src)
    for x in range(len(gt_filenames)):
        gt_filename=gt_filenames[x]
        print("file name: ",gt_filename)
        img_name=filter(str.isdigit,gt_filename)+'.jpg'
        img=cv2.imread(img_db+img_name)

        sh=str(img.shape[0])
        sw=str(img.shape[1])
        txt2xml(gt_src,gt_filename,gt_dst,sw,sh)

if __name__ == '__main__':
    gt_src='E:/data/ocr/preprocessing/HUST-TR400/label/'
    gt_dst='E:/data/ocr/preprocessing/HUST-TR400/label_xml/'
    img_db='E:/data/ocr/preprocessing/HUST-TR400/image/'
	
    main(gt_src,gt_dst,img_db)

