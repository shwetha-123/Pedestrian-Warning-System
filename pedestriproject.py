#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:43:09 2020

@author: shwetha
"""
import numpy as np
import cv2
import tensorflow as tf
import urllib.request

class DetectorAPI:
    def __init__(self,path_to_ckpt):
        self.path_to_ckpt=path_to_ckpt
        self.detection_graph=tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def=tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt,'rb')as fid:
                serialized_graph=fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def,name='')
        self.defult_graph=self.detection_graph.as_default()
        self.sess=tf.Session(graph=self.detection_graph)
        self.image_tensor=self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes=self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores=self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes=self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections=self.detection_graph.get_tensor_by_name('num_detections:0')
    def ProcessFrame(self,image):
           image_np_expanded=np.expand_dims(image,axis=0)
           (boxes,scores,classes,num)=self.sess.run([self.detection_boxes,self.detection_scores,self.detection_classes,self.num_detections],feed_dict={self.image_tensor:image_np_expanded})
       
           
           im_height,im_width,_=image.shape
           boxes_list=[None for i in range(boxes.shape[1])]
           for i in range(boxes.shape[1]):
               boxes_list[i]=(int(boxes[0,i,0]*im_height),int(boxes[0,i,1]*im_width),int(boxes[0,i,2]*im_height),int(boxes[0,i,3]*im_width))
    
   
           return boxes_list,scores[0].tolist(),[int(x)for x in classes[0].tolist()],int(num[0])
    def close(self):
      self.sess.close()
      self.default_graph.close()
if __name__ == "__main__":
    path="/home/shwetha/Downloads/pedistrians/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
    a=DetectorAPI(path_to_ckpt=path)
    threshold=0
    cap=cv2.VideoCapture(0)
    baseurl=" https://api.thingspeak.com/update?api_key=J3A2LXVRW5SVKRQ1&field1=0"
    while True:
        ret,img=cap.read()
        boxes,scores,classes,num=a.ProcessFrame(img)
        for i in range(len(boxes)):
            if classes[i]==1:
               boo=True
               print("found")
               
               box=boxes[i]
               cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            else:
                boo=False
                print("not found")
            f=urllib.request.urlopen(baseurl+str(boo))
            f.read()
            f.close()
            
                
        cv2.imshow("Window1",img)
        if((cv2.waitKey(1) & 0xFF)== ord('q')):
            break
cap.release()
cv2.destroyAllWindows()

