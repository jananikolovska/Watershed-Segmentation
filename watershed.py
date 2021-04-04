#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


# In[4]:


def thresholded(img):
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = np.invert(thresh)
    return thresh


# In[16]:


def count(img,draw = False):
    plt.figure(figsize=(8, 8))
    thresh = thresholded(img)
  
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    eroded=cv2.erode(opening,kernel,iterations=10)
    dilated=cv2.dilate(eroded,kernel,iterations=6)
  
  
    dist_transform = cv2.distanceTransform(dilated,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
  
    sure_fg = np.uint8(sure_fg)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    sure_fg=cv2.erode(sure_fg,kernel,iterations=5)
    sure_fg=cv2.dilate(sure_fg,kernel,iterations=5)

    output = cv2.connectedComponentsWithStats(sure_fg, cv2.CV_32S)
  
    centroids = output[3]
    centroids_final=[]
  
    for i in range(1,len(centroids)):
        centroids_final.append(centroids[i])
    centroids_final=np.array(centroids_final)
    print(f"Number of Nuclei: {len(centroids_final)}")
  
    plt.imshow(resized_original,cmap='gray')
    plt.title("Result")
    plt.axis("off")
    plt.scatter(centroids_final[:,0],centroids_final[:,1])
    plt.savefig("result.png",dpi=100)
    return centroids_final


# In[19]:


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="insert filepath to image")
args = parser.parse_args()
original = cv2.imread(args.filename,cv2.IMREAD_GRAYSCALE)
resized_original = cv2.resize(original,(1024,1024))
centroids = count(resized_original,draw=True)


# In[ ]:




