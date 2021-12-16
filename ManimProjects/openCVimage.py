import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class GrayScaleImage:
    
    def imageConstruct(self):
        sns.set(color_codes=True)
        image = cv2.imread('demoCartoon.png')
        imageCV2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.imshow(imageCV2)
        plt.show()
        
        print(image.shape)
        print(imageCV2.shape)
        data = np.array(imageCV2)
        flattened = data.flatten()
        print(flattened.shape)


