# -*- coding: UTF-8 -*-
import cv2
import sys
import shutil
reload(sys)
sys.setdefaultencoding('utf-8')

with open('predict_result.txt','r') as f:
	for line in f.readlines():
		try:
			name,tag = line.decode('utf-8').strip().split()
		except:
			name = line.decode('utf-8').strip().split()[0]
			tag = 'null'
		ex = name.split('.')[1]
		shutil.copyfile('../images/'+name,'./result_images/'+tag.decode('utf-8')+'.'+ex)


		
