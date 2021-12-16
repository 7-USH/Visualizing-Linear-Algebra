from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import os
from manimlib.imports import *
from manim import *


class EigenFace(Scene):
    def construct(self):
        dir = r'lfwcrop_grey\faces'
        celebrity_photos = os.listdir(dir)[1:1001]
        celebrity_images = [dir+'/' + photo for photo in celebrity_photos]
        self.images = np.array([plt.imread(image)
                           for image in celebrity_images], dtype=np.float64)
        celebrity_names = [name[:name.find('0')-1].replace("_", " ")
                           for name in celebrity_photos]
        n_samples, h, w = self.images.shape
        self.plot_portraits(self.images, celebrity_names, h, w, n_row=4, n_col=4)
        self.clear()

        P, C, M, Y = self.pca(n_samples,h,w, 1000)
        eigenfaces = C.reshape((1000, h, w))
        
        bigGroup = Group()
        for i in range(4 * 4):
            window = ImageMobject(eigenfaces[i].reshape((h, w)))
            window.scale(0.6)
            bigGroup.add(window).arrange(direction=RIGHT)
        bigGroup.arrange_in_grid(4, 4)
        self.play(FadeIn(bigGroup))
        self.wait()
      
    def plot_portraits(self, images, titles, h, w, n_row, n_col):
        plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
        bigGroup = Group()
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row * n_col):
            window = ImageMobject(images[i].reshape((h, w)))
            window.scale(0.6)
            bigGroup.add(window).arrange(direction=RIGHT)
        bigGroup.arrange_in_grid(4,4)         
        self.play(FadeIn(bigGroup))
        self.wait()    
         
    def pca(self,n,h,w, n_pc):
        X = self.images.reshape(n,h*w)
        mean = np.mean(X, axis=0)
        centered_data = X-mean
        U, S, V = np.linalg.svd(centered_data)
        components = V[:n_pc]
        projected = U[:, :n_pc]*S[:n_pc]
        return projected, components, mean, centered_data
