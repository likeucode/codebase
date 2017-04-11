#############################
#author : ke.liu@xiaoi
#date   : 04/11/2017
#usage  : convert text to xml for icdar2013 dataset
#############################
import os
from xml.dom.minidom import Document

def txt2xml(gt_src,gt_filename,gt_dst,sw,sh):

	doc = Document()  
	annotation = doc.createElement('annotation') 
	doc.appendChild(annotation)

	folder = doc.createElement('folder') 
	
	folder.appendChild(doc.createTextNode('Challenge2_ICDAR2013_Task12')) 
	filename = doc.createElement('filename') 
	#filename.appendChild(doc.createTextNode(filter(str.isdigit,gt_filename)+'.jpg'))
	filename.appendChild(doc.createTextNode('img_'+filter(str.isdigit,gt_filename)+'.jpg'))##test
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
		#label=labelLines[i].split(' ') ##train
		label=labelLines[i].split(',')##test
		objectNode=doc.createElement('object')

		nameNode=doc.createElement('name')
		#nameNode.appendChild(doc.createTextNode(label[-1][1:-1]))##train
		#nameNode.appendChild(doc.createTextNode(label[-1][2:-1]))##test
		nameNode.appendChild(doc.createTextNode('text'))

		bdnboxNode=doc.createElement('bndbox')
		xminNode=doc.createElement('xmin')
		xminNode.appendChild(doc.createTextNode(label[0]))
		yminNode=doc.createElement('ymin')
		yminNode.appendChild(doc.createTextNode(label[1]))
		xmaxNode=doc.createElement('xmax')
		xmaxNode.appendChild(doc.createTextNode(label[2]))
		ymaxNode=doc.createElement('ymax')
		ymaxNode.appendChild(doc.createTextNode(label[3]))

		bdnboxNode.appendChild(xminNode)
		bdnboxNode.appendChild(yminNode)
		bdnboxNode.appendChild(xmaxNode)
		bdnboxNode.appendChild(ymaxNode)

		objectNode.appendChild(nameNode)
		objectNode.appendChild(bdnboxNode)

		annotation.appendChild(objectNode)

	fp = open(gt_dst+filter(str.isdigit,gt_filename)+'.xml', 'w')
	doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
	fp.close()

def main(gt_src,gt_dst,img_size_file):
	with open(img_size_file) as sf:
		label_sizes=sf.read()
	size_lines=label_sizes.splitlines()
	file_list=[size_lines[i].split()[0] for i in range(len(size_lines))]

	gt_filenames=os.listdir(gt_src)
	for x in range(len(gt_filenames)):
		gt_filename=gt_filenames[x]
		print("file name: ",gt_filename)
		img_file_name=filter(str.isdigit,gt_filename)
		#idx=file_list.index(img_file_name)##train
		idx=file_list.index('img_'+img_file_name)##test
		sh=size_lines[idx].split()[1]
		sw=size_lines[idx].split()[2]
		txt2xml(gt_src,gt_filename,gt_dst,sw,sh)

if __name__ == '__main__':
	gt_src='/home/ke/ocr/TextBoxes/data/icdar/icdar2013/task1_gt/'
	gt_dst='/home/ke/ocr/TextBoxes/data/icdar/icdar2013/task1_gt_xml/'
	img_size_file='/home/ke/ocr/TextBoxes/data/icdar/icdar2013/train_test_list/test_name_size.txt'
	main(gt_src,gt_dst,img_size_file)

