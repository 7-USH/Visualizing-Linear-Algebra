from re import U
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import os
from manimlib.imports import *
from manim import *
import cv

class EigenFace(Scene):
    def construct(self):
        dir = r'lfwcrop_grey\faces'
        celebrity_photos = os.listdir(dir)[1:1001]
        celebrity_images = [dir+'/' + photo for photo in celebrity_photos]
        self.images = np.array([plt.imread(image)
                           for image in celebrity_images], dtype=np.float64)   
        n_samples, h, w = self.images.shape
        
        resultTex = TextMobject(
            "We have gathered some of the famous Celebrity images ")
   
        self.play(Write(resultTex))
        self.wait()
        self.clear()


        result = self.plot_portraits(self.images, h, w, n_row=4, n_col=4)
        self.play(FadeIn(result))
        self.wait(2)

        X = self.images.reshape(n_samples, h*w)
        P, C, M, Y = self.pca(X, 1000)

        eigenfaces = C.reshape((1000, h, w))

        plt.figure(figsize=(2.2 *4, 2.2 *4))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(4* 4):
            plt.subplot(4, 4, i + 1)
            plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
        plt.savefig("output.png")
    
        meanImage = ImageMobject(M.reshape(h,w))
        groupy = Group()
        groupy.add(meanImage)
        self.play(Transform(result,groupy))
        self.wait()

        meanText = TextMobject("Mean of all the images")
        meanText.next_to(groupy,direction=TOP)
        self.play(Write(meanText))
        self.wait()

        self.clear()
    
        eigenFaceText = VGroup(
            TextMobject("Using the mean image we are able"),
            TextMobject("to achieve eigen Faces of all images")).arrange(direction=DOWN)
        self.play(Write(eigenFaceText))
        self.wait()
        self.clear()

        self.convertImage("output.png")
        eigenFaceImages = ImageMobject("eigenFaces.png")
        eigenFaceImages.scale(3)   
        self.play(FadeIn(eigenFaceImages))
        self.wait(3)
        self.clear()

        recovered_images = [self.reconstruction(
            Y, C, M, h, w, i) for i in range(len(self.images))]

        recoverGroup = TextMobject("We can recover these Images")
        self.play(Write(recoverGroup))
        self.wait()
        self.clear()

        recoveredResult = self.plot_portraits(recovered_images,h,w,4,4)
        self.play(FadeIn(recoveredResult))
        self.wait(2)    
  
    def plot_portraits(self, images,h, w, n_row, n_col):
        plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
        bigGroup = Group()
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row * n_col):
            window = ImageMobject(images[i].reshape((h, w)))
            window.scale(0.6)
            bigGroup.add(window).arrange(direction=RIGHT)
        bigGroup.arrange_in_grid(4,4)         
        return bigGroup    
         
    def pca(self,X, n_pc):
        n_samples, n_features = X.shape
        mean = np.mean(X, axis=0)
        centered_data = X-mean
        U, S, V = np.linalg.svd(centered_data)
        components = V[:n_pc]
        projected = U[:, :n_pc]*S[:n_pc]
        return projected, components, mean, centered_data

    def reconstruction(self,Y, C, M, h, w, image_index):
        weights = np.dot(Y, C.T)
        centered_vector = np.dot(weights[image_index, :], C)
        recovered_image = (M+centered_vector).reshape(h, w)
        return recovered_image

    def convertImage(self,image):
        img = Image.open(image)
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for items in datas:
            if items[0] == 255 and items[1] == 255 and items[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(items)
        img.putdata(newData)
        img.save("./eigenFaces.png", "PNG")
        print("Successful")

    
