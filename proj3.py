########################################################################
## File:    proj3.py                                                  ##
## Author:  Anna Aladiev                                              ## 
## Date:    05/01/2016                                                ##
## Course:  CMSC 471 - Artificial Intelligence (Spring 2016)          ##
## Section: 02                                                        ##
## E-mail:  aladiev1@umbc.edu                                         ## 
##                                                                    ##
##   This file contains the python code for PROJECT 3.                ##
## This program classifies a given image as a smile, hat, dollar,     ##                                                         
## hash (octothorpe), or heart. From the command prompt, please type  ##
## python proj3.py <filepath> .                                       ##                                                                                                        
##                                                                    ##                                                                  
########################################################################

import sys

from sklearn import svm
from skimage import io
from skimage.color import rgb2gray

import numpy as np
import os

#converts image to grayscale and returns it as an np array
def convImage(filepath):

    image = io.imread(filepath)
    gray_image = rgb2gray(image)
    binary_image = np.where(gray_image > np.mean(gray_image), 1.0, 0.0)

    return binary_image

#creates a support vector machine from the training data
def createSVM():

    data = []
    classification = []    

    for i in range(0, 100):
        number = ""
        if(i < 10):
            number = "0" + str(i)
        else:
            number = str(i)
        
        smilePath = "Training Data/smile/" + number + ".jpg"
        hatPath = "Training Data/hat/" + number + ".jpg"
        hashPath = "Training Data/hash/" + number + ".jpg"
        heartPath = "Training Data/heart/" + number + ".jpg"
        dollarPath = "Training Data/dollar/" + number + ".jpg"
        
        if(os.path.exists(smilePath)):
            smile = convImage(smilePath)
            data.append(smile)
            classification.append(1)
            
            
        if(os.path.exists(hatPath)):
            hat = convImage(hatPath)
            data.append(hat)
            classification.append(2)
            
            
        if(os.path.exists(hashPath)):
            hashtag = convImage(hashPath)
            data.append(hashtag)
            classification.append(3)
            
            
        if(os.path.exists(heartPath)):
            heart = convImage(heartPath)
            data.append(heart)
            classification.append(4)
            
            
        if(os.path.exists(dollarPath)):
            dollar = convImage(dollarPath)
            data.append(dollar)
            classification.append(5)
            
            
    #shape data
    n_samples = len(data)
    shaped_data = np.asarray(data).reshape((n_samples, -1))
    
    #create estimator
    clf = svm.LinearSVC()
    
    #learn from data
    clf.fit(shaped_data, classification)

    return clf

def userImage():
    
    if(len(sys.argv) != 2):
        print("Invalid input.")
        print("Please type in: python proj3.py <filepath>")
        print("<filepath> has a jpg extension")
        
        
    filepath = sys.argv[1]


    if(not os.path.exists(filepath)):
        print("The file does not exist.")
        exit()
        
    return(filepath)  

def main():

    #filepath = userImage()    
    
    SVM = createSVM()

    image = convImage(filepath)
    image_array = image.reshape(1, -1)

    #predict outcome
    pNumber = SVM.predict(image_array)

    if(pNumber[0] == 1):
        p = "Smile"
    elif(pNumber[0] == 2):
        p = "Hat"
    elif(pNumber[0] == 3):
        p = "Hash"
    elif(pNumber[0] == 4):
        p = "Heart"
    elif(pNumber[0] == 5):
        p = "Dollar"

    print(p)

main()