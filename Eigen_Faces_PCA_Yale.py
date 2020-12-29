#import
import turicreate as tc
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.cm as cm

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from numpy.linalg import eig
from sklearn.svm import SVC
%matplotlib inline
paper_referred = 'https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71'

image_path = '../Linear_Algebra/Face Detection_PCA/ATT/'
#image_path = '../Linear_Algebra/Face Detection_PCA/YALE/faces'

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles
files = getListOfFiles(image_path)
files.sort()
files = files[1:]
len(files)


import glob
from os import path
'''
# Read the image filespatttern
filespatttern = '*.pgm'
pathPattern = path.join(image_path,filespatttern)
files = glob.glob(pathPattern)
'''

# Convert each [m x n] image matrix into [(m x n) x 1] image vector.(Flatten image matrix to vector).
len(files)
images =[]
for image in files:
    #im = Image.open(image).convert('L')
    im = Image.open(image)
    images.append(np.asarray(im))

flattened_images = []
for i in range(0, len(images)):
    flattened_images.append(np.reshape(images[i], (images[0].shape[0] * images[0].shape[1])))

flattened_images = np.asmatrix(flattened_images)

flattened_images.shape

images[0].shape
# Find average_face_vector, sum(all image vectors)/number(images)

mean_face = np.mean(flattened_images, axis=0)
mean_face
mean_face.shape

plt.subplot(121), plt.imshow(np.reshape(mean_face, (images[0].shape[0],images[0].shape[1])), cmap="gray"), plt.axis('off'), plt.title('Mean face')



# Subtract average_face_vector from every image vector.
px_images = flattened_images - mean_face

# Stack all (image_vectors — average_face_vector) in matrix forming A = [(m x n) x i] matrix (where i = number of images).
px_images.shape

# Calculate covariance matrix of above matrix -> C = A*transpose(A)
covariance_matrix = np.dot(px_images, px_images.T)
#covariance_matrix = covariance_matrix/len(files)
covariance_matrix.shape
covariance_matrix
val, vec = np.linalg.eigh(covariance_matrix)
prod = np.dot(px_images.T, vec)
#for i in range(px_images.shape[0]):
#    prod[:,i] = prod[:,i]/np.linalg.norm(prod[:,i])

#prod.shape
#prod
val.shape
vec.shape
prod.shape
idx = np . argsort ( - val )
val = val [ idx ]
prod = prod [: , idx ]

prod.shape
prod[0]
for i in range(len(images)):
    prod[:,i] = prod[:,i]/np.linalg.norm(prod[:,i])
prod[0]

def plot_eigvals():
    plt.plot(val, 'r')
    plt.title('eigenValues Ordered')
    plt.ylabel('Value of eigenValues')
    plt.xlabel('index number')

plot_eigvals()
plt.plot(np.cumsum(val), 'g'), plt.xlabel('number of components'), plt.ylabel('cumulative explained variance'), plt.show()

prod.shape
plt.subplot(121), plt.imshow(np.reshape(prod[:,0], (images[0].shape[0],images[0].shape[1])), cmap="gray"), plt.subplot(1,2,2), plt.imshow(np.reshape(prod[:,7], (images[0].shape[0],images[0].shape[1])), cmap="gray")

#eig_f = np.dot(p2_images.T, eigenVectors)

def show_EigenVecs():
    fig, axes = plt.subplots(20, 20, figsize=(6, 6), subplot_kw={'xticks':[], 'yticks':[]})
    #print(axes.flat)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.reshape(np.array(prod)[:,i], (images[0].shape[0], images[0].shape[1])), cmap = 'gray')


show_EigenVecs()


prod.shape
px_images.shape
weights = np.dot(px_images, prod)
weights.shape
px_images[0].reshape(1,-1).shape
px_images[0].shape

reconstructed_flattened_image_vector = mean_face + np.dot(weights, prod.T)

def show_reconstructed_images(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(20, 20, figsize=(20, 20), subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.reshape(np.array(pixels)[i], (images[0].shape[0], images[0].shape[1])) , cmap='gray')
	plt.show()


show_reconstructed_images(reconstructed_flattened_image_vector)

plt.subplot(121), plt.imshow(np.reshape(reconstructed_flattened_image_vector[55], (images[0].shape[0],images[0].shape[1])), cmap="gray"), plt.subplot(1,2,2), plt.imshow(np.reshape(flattened_images[90], (images[0].shape[0],images[0].shape[1])), cmap="gray")

K = 50
selected_eigenVectors = prod[:, 0:K]

selected_eigenVectors.shape

selected_weights = np.dot(px_images, selected_eigenVectors)
selected_weights.shape
reconstructed_flattened_image_vector_k = mean_face + np.dot(selected_weights, selected_eigenVectors.T)
reconstructed_flattened_image_vector_k.shape


show_reconstructed_images(reconstructed_flattened_image_vector_k)



# To see progress

def reconstruction_progress(k):

    selected_eigenVectors = prod[:, 0:k]
    selected_weights = np.dot(px_images, selected_eigenVectors)
    reconstructed_flattened_image_vector_k = mean_face + np.dot(selected_weights, selected_eigenVectors.T)
    return reconstructed_flattened_image_vector_k

K = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120]
E = []
for k in K:
    E.append(reconstruction_progress(k)[21])
    #E.append(np.reshape(reconstruction_progress(k)[0], (300, 300), 'F'))



def show_reconstructed_images_progress(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(2, 5, figsize=(15, 15))
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.reshape(np.array(pixels)[i], (images[0].shape[0], images[0].shape[1])) , cmap='gray')
        #ax.set_title('K = ' + str(i))
    #fig.set_facecolor('w')
    #plt.tight_layout()
    #plt.show()




show_reconstructed_images_progress(E)
