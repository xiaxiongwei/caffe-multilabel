import math
import numpy as np
import sys
import cv2
caffe_root='./'
sys.path.insert(0,caffe_root+'python')

imgname=sys.argv[1]
test_img_path = '/mnt/lvmhdd/xiaxiongwei/workspace/tagging/multilabel_caffe/{}'.format(imgname)

batch_size = 50

test_images_list = open(test_img_path,'r').readlines()
image_num = len(test_images_list)

model_def='/mnt/lvmhdd/xiaxiongwei/workspace/tagging/multilabel_caffe/examples/coco_caption/vgg16.deploy.prototxt'
#model_def='/mnt/lvmhdd/xiaxiongwei/workspace/tagging/multilabel_caffe/examples/coco_caption/gnet.deploy.prototxt'
#model_weight='/mnt/lvmhdd/xiaxiongwei/models/coco_img_captioning/tagging_gnet_multilabel_235_adam_iter_80000.caffemodel'
model_weight='/mnt/lvmhdd/xiaxiongwei/models/coco_img_captioning/tagging_vgg16_multilabel_235_uniform_adam_iter_51000.caffemodel'

import caffe


def get_class_list():
	class_list=[]
	with open('../class_list.txt','r') as f:
		for line in f.readlines():
			tag = line.strip()
			class_list.append(tag)
	return class_list


class_list = get_class_list()

caffe.set_mode_gpu()
caffe.set_device(2)
net = caffe.Net(model_def,model_weight,caffe.TEST)

net.blobs['data'].reshape(batch_size,3,224,224)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})

transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.array([104,117,123]))
#transformer.set_raw_scale('data',255)
#transformer.set_channel_swap('data',(2,1,0))


def test_image():
	iter_num = int(math.ceil(image_num/batch_size))
	print "batchsize:{}, total test images:{}, iter_num:{}".format(batch_size,image_num,iter_num)
	output = []
	for i in xrange(iter_num):
		transformed_image = []
		for line in  test_images_list[i*batch_size:(i+1)*batch_size]:
			filename = line.strip().split('\t')[0]
			image=cv2.imread('../images/{}'.format(filename))
			image = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
			print '../images/{}'.format(filename)
			transformed_image.append(transformer.preprocess('data',image))
		net.blobs['data'].data[...] = transformed_image
		output.extend(net.forward()['prob'])
	return output

def print_result(output,class_list):
	fout=open('predict_result.txt','w')
	with open(test_img_path,'r') as f:
		img_cnt = 0
		for line in test_images_list:
			predicted = []
			filename, tags = line.strip().split('\t')
			output_prob = output[img_cnt]

			for i in [(class_list[i], output_prob[i]) for i in range(len(output_prob)) if output_prob[i]>=0.05]:
				predicted.append([i[0],i[1]])

			predicted = sorted(predicted,key=lambda d:d[1],reverse=True)

			fout.write(filename+" ")
			#fout.write("ground truth : "+tags+"\n")
			#fout.write("predicted : "+" ".join([i[0]+'('+str(i[1])+')' for i in predicted])+"\n")
			fout.write("_".join([i[0]  for i in predicted])+"\n")
			#fout.write("====================================="+"\n")
			img_cnt+=1
	fout.close()


print_result(test_image(), class_list)
