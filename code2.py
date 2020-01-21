import sys
import numpy as np
from PIL import Image
import os
import glob
import cv2
import matplotlib.pyplot as plt

#number of training examples
m_happy=104
m_sad=100
m_test=25
lt=100
bt=100

#input array
#coloumn wise arrangement
input_images_happy=np.zeros((lt*bt,m_happy))
input_images_sad=np.zeros((lt*bt,m_sad))
input_images_test=np.zeros((lt*bt,m_test))


############################################################################################################################
#preprocess the images into a vectorized format
img_dir_happy = "/home/dell/Desktop/machineLearning/Expression2/train_set/happy" # Enter Directory of all images 
img_dir_sad = "/home/dell/Desktop/machineLearning/Expression2/train_set/sad" # Enter Directory of all images 
img_dir_test = "/home/dell/Desktop/machineLearning/Expression2/test_set/" # Enter Directory of all images 
data_path_happy = os.path.join(img_dir_happy,'*')
data_path_sad = os.path.join(img_dir_sad,'*')
data_path_test = os.path.join(img_dir_test,'*')
files_happy = glob.glob(data_path_happy)
files_sad = glob.glob(data_path_sad)
files_test = glob.glob(data_path_test)

i=0
for f1 in files_happy:
 img = Image.open(f1)
 temp=np.array(img)
 #cv2.imshow("ex",temp)
 #cv2.waitKey()
 temp=cv2.resize(temp,(lt,bt))
 #Grayscaling operation
 temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
 #print(temp)
 #cv2.imshow("ex",temp)
 #cv2.waitKey()
 #flatenning the images so that we can stack it coloumn wise
 input_images_happy[:,i]=temp.ravel()
 i+=1

i=0
for f1 in files_sad:
 img = Image.open(f1)
 temp=np.array(img)
 #cv2.imshow("ex",temp)
 #cv2.waitKey()
 temp=cv2.resize(temp,(lt,bt))
 #Grayscaling operation
 temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
 #print(temp)
 #cv2.imshow("ex",temp)
 #cv2.waitKey()
 #flatenning the images so that we can stack it coloumn wise
 input_images_sad[:,i]=temp.ravel()
 i+=1
i=0


 #loading test data
for f1 in files_test:
 img = Image.open(f1)
 temp=np.array(img)
 #cv2.imshow("ex",temp)
 #cv2.waitKey()
 temp=cv2.resize(temp,(lt,bt))
 #Grayscaling operation
 temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
 #print(temp)
 cv2.imshow("ex",temp)
 cv2.waitKey()
 #flatenning the images so that we can stack it coloumn wise
 input_images_test[:,i]=temp.ravel()
 i+=1
i=0

############################################################################################################################


############################################################################################################################
#Starting to compute the mean and normalize the data
average_happy=np.sum(input_images_happy,axis=1,keepdims=True)/m_happy
average_sad=np.sum(input_images_happy,axis=1,keepdims=True)/m_sad
cv2.imshow("happy",np.array(average_happy.reshape((bt,lt)),dtype = np.uint8))
cv2.imshow("sad",np.array(average_sad.reshape((bt,lt)),dtype = np.uint8))
cv2.waitKey()
input_im_norm_happy = input_images_happy - average_happy
input_im_norm_sad = input_images_sad - average_sad

input_im_norm_happy_test = input_images_test - average_happy
input_im_norm_sad_test = input_images_test - average_sad
############################################################################################################################


############################################################################################################################
#Calculating the eigenvalues and eigenvectors i.e. eigenfaces
print("Calculating PCA ", end="...")
mean_happy, eigen_happy = cv2.PCACompute(input_images_happy,mean=None,maxComponents=m_happy)
mean_sad, eigen_sad = cv2.PCACompute(input_images_sad,mean=None,maxComponents=m_sad)
print()
correct_eigen_happy=np.dot(input_im_norm_happy,eigen_happy)
correct_eigen_sad=np.dot(input_im_norm_sad,eigen_sad)

print(correct_eigen_happy.shape)
print(correct_eigen_sad.shape)

eigen_norm_happy=np.zeros((lt*bt,m_happy),dtype=float)
eigen_norm_sad=np.zeros((lt*bt,m_sad),dtype=float)
for j in range(m_happy):
 l=np.linalg.norm(correct_eigen_happy[:,j])
 eigen_norm_happy[:,j]=correct_eigen_happy[:,j]/l
for j in range(m_sad):
 l=np.linalg.norm(correct_eigen_sad[:,j])
 eigen_norm_sad[:,j]=correct_eigen_sad[:,j]/l

print("Check after normalization ")
print(eigen_norm_happy.shape)
print(eigen_norm_sad.shape)
print("computed")

############################################################################################################################

############################################################################################################################
#Computing the weights for all images
temp_eigen_happy=eigen_norm_happy.reshape((m_happy,lt*bt))
temp_eigen_sad=eigen_norm_sad.reshape((m_sad,lt*bt))

weight_happy=np.zeros((m_happy,m_happy))
weight_sad=np.zeros((m_sad,m_sad))

for k in range(m_happy):
 for j in range(m_happy):
  weight_happy[j,k]=np.dot(temp_eigen_happy[j],input_im_norm_happy[:,k])

for k in range(m_sad):
 for j in range(m_sad):
  weight_sad[j,k]=np.dot(temp_eigen_sad[j],input_im_norm_sad[:,k])
print(weight_happy.shape)
print(weight_sad.shape)


############################################################################################################################

#testing phases
weight_happy_test=np.zeros((m_happy,m_test))
weight_sad_test=np.zeros((m_sad,m_test))
for k in range(m_test):
 for j in range(m_happy):
  weight_happy_test[j,k]=np.dot(temp_eigen_happy[j],input_im_norm_happy_test[:,k])

for k in range(m_test):
 for j in range(m_sad):
  weight_sad_test[j,k]=np.dot(temp_eigen_sad[j],input_im_norm_sad_test[:,k])

print(weight_happy_test.shape)
print(weight_sad_test.shape)
print(eigen_norm_happy[:,0].shape)
for i in range(m_test):
 #happy check
 happy_distances=np.zeros((1,m_happy))
 sad_distances=np.zeros((1,m_sad))
 for g in range(m_happy):
  diff=weight_happy[:,g]-weight_happy_test[:,i]
  hap=(np.linalg.norm(diff))
  happy_distances[0,g]=hap
 hap_val=np.amin(happy_distances)
 #sad check
 for g in range(m_sad):
  diff=weight_sad[:,g]-weight_sad_test[:,i]
  sad=(np.linalg.norm(diff))
  sad_distances[0,g]=sad
 sad_val=np.amin(sad_distances)
 print(hap_val)
 print(sad_val)
 if(hap_val<sad_val):
  print("Happy")
 else:
  print("Sad")
 cv2.imshow("image",np.array(input_images_test[:,i].reshape((100,100)),dtype=np.uint8))
 cv2.waitKey()
 print("############################")
print("computed")
"""
temp=np.zeros((77760,1))
for j in range(m):
 t=eigen_norm[:,j].reshape((77760,1))
 temp+=weight[j]*t
dist1 = cv2.normalize(temp, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
dist1=dist1.reshape((320,243))
cv2.imshow("v",np.array(dist1,dtype=np.uint8))
cv2.imwrite("/home/dell/Desktop/machineLearning/Expression/new1111.png",dist1)
temper=temp.reshape((320,243))
val=np.array(temper,dtype=np.uint8)
cv2.imshow("img",np.array(temper,dtype=np.uint8))
cv2.imwrite("/home/dell/Desktop/machineLearning/Expression/test1.png",np.array(temper,dtype=np.uint8))
cv2.imshow("i",np.array(input_im_norm[:,0].reshape((320,243)),dtype=np.uint8))
cv2.waitKey()
"""
