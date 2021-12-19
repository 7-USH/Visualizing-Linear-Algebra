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
        n_samples, h, w = self.images.shape
        celebrity_names = [
            name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
        originalImages = self.images
        resultText = VGroup(TextMobject("We gathered the potraits of", " 1000 ", "people"), TextMobject(
            "for the facial recognition using Eigen Faces"))
        resultText.arrange(direction=DOWN).scale(0.7)
        resultText[0][1].set_color(RED)

        # self.play(Write(resultText))
        self.wait()
        self.play(Write(resultText))
        self.wait(5)
        self.clear()
        result2 = self.plot_portraits2(
            originalImages,  h, w, n_row=4, n_col=4)
        celebrityText = TextMobject(
            "Here are ", " 16 ", " of them", direction=TOP)
        celebrityText[1].set_color(RED)
        result3 = self.plot_portraits2(
            originalImages,  h, w, n_row=4, n_col=4)
        result = self.plot_portraits(
            self.images,  h, w, n_row=4, n_col=4)

        #result.next_to(celebrityText, direction=DOWN, buff=2)
        self.play(Write(celebrityText))
        self.wait()
        self.clear()
        self.play(FadeIn(result))

        self.wait(3)
        n_components = 50
        X = self.images.reshape(n_samples, h*w)
        P, C, M, Y = self.pca(X, n_components)

        eigenfaces = C.reshape((n_components, h, w))

        plt.figure(figsize=(2.2 * 4, 2.2 * 4))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(4 * 4):
            plt.subplot(4, 4, i + 1)
            plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
        plt.savefig("output.png")

        meanImage = ImageMobject(M.reshape(h, w))
        groupy = Group()
        groupy.add(meanImage)
        self.play(Transform(result, groupy))
        self.wait()

        meanText = TextMobject("Mean of all the images")
        meanText.next_to(groupy, direction=TOP)
        self.play(Write(meanText))
        self.wait()

        self.clear()

        sampleImage = ImageMobject('David_Beckham_0001.png')
        sampleImage.scale(0.7)
        sampleImage.shift(LEFT*2)
        brace1 = Brace(sampleImage, direction=TOP)

        braceText1 = brace1.get_text("64px")
        braceText1.scale(0.7)
        brace2 = Brace(sampleImage, direction=LEFT)

        braceText2 = brace2.get_text("64px")
        braceText2.scale(0.7)

        sampleImageText = TextMobject(
            "This image can be represented as a", " Column Matrix", " as")
        sampleImageText[1].set_color(GREEN)
        sampleImageText.move_to(TOP*0.7).scale(0.8)
        equalsText = TextMobject("=")

        text = TexMobject(
            "\\begin{bmatrix} {a}{01}  \\\ {a}{02}  \\\ . \\\ . \\\ . \\\ .\\\ . \\\ {a}{63} \\\ {a}{64}  \\end{bmatrix}"
            #"\\begin{bmatrix} {b}{11} & {b}{12} & . & . & {b}{1n} \\\ {b}{21} & {b}{22} & . & . & {b}{2n} \\\ . & . & . & . & . \\\ {b}{m1} & {b}{m2} & . & . & {b}_{mn}\\end{bmatrix}", "=",
        )
        text.scale(0.5).move_to(RIGHT)

        newText = TextMobject(
            "Thus a set of ", " 1000 ", " images can be represented as: ")
        newText[1].set_color(RED)
        newText.move_to(TOP*0.7).scale(0.8)
        matrixUpperText1 = TextMobject("X \\quad = ")
        matrixUpperText1.shift(LEFT*2).scale(0.8)

        matrixText2 = TexMobject(
            """ \\begin{bmatrix} {a}{11} & {a}{12} & {a}{13} & . & . & ........\\\ {a}{21} & {a}{22} & {a}{23} & . & . & ........ \\\ {a}{31} & {a}{32} & {a}{33} & . & . & ........    
            \\\ . & . & . & . & . & ........\\\ . & . & . & . & . & ........\\\ . & . & . & . & . & ........\\\ . & . & . & . & . & ........
            \\\ . & . & . & . & . & ........\\\ . & . & . & . & . & ........\\\ {a}{n1}   & . & . & . & . & ........\\end{bmatrix}""")
        matrixText2.scale(0.5).shift(RIGHT)

        matrixUpperText2 = TextMobject(
            "I1 \\quad I2 \\quad I3 .  .  .  . ")
        matrixUpperText2.set_color(YELLOW)
        matrixUpperText2.scale(0.58).shift(0.70*RIGHT).shift(0.45*TOP)

        self.play(FadeIn(sampleImage))
        self.wait(1)

        self.play(GrowFromCenter(brace1), FadeIn(braceText1),
                  GrowFromCenter(brace2), FadeIn(braceText2),)
        self.wait()

        self.play(Write(sampleImageText))
        self.wait(1)

        self.play(Write(equalsText), Write(text))
        self.wait(5)
        self.clear()

        self.play(Write(newText))
        self.wait()

        self.play(Write(matrixUpperText1), Write(matrixText2))
        self.wait()
        self.wait()
        self.play(Write(matrixUpperText2))
        self.wait(5)
        self.clear()

        eigenFaceText = VGroup(
            TextMobject("We subtract", " X ", " from the"),
            TextMobject("Mean image to get centered image matrix")).arrange(direction=DOWN)
        eigenFaceText2 = VGroup(TextMobject(
            "On this centered image, we perform "), TextMobject(
            "Singular Value Decomposition", " to obtain eigenfaces")).arrange(direction=DOWN)
        eigenFaceText.shift(TOP*1.5).scale(0.7)
        eigenFaceText[0][1].set_color(GREEN)
        eigenFaceText2[1][1].set_color(GREEN)
        eigenFaceText2.shift(DOWN*1.5).scale(0.7)
        self.play(Write(eigenFaceText))
        self.wait(3)
        self.play(Write(eigenFaceText2))
        self.wait(3)
        self.clear()

        eigenFaceText3 = TextMobject("Thus the Eigen Faces obtained:")
        eigenFaceText3.move_to(3.5*UP)
        self.play(Write(eigenFaceText3))
        self.convertImage("output.png")
        eigenFaceImages = ImageMobject("eigenFaces.png")
        eigenFaceImages.scale(3)
        self.play(FadeIn(eigenFaceImages))
        self.wait(3)
        self.clear()

        recovered_images = [self.reconstruction(
            Y, C, M, h, w, i) for i in range(len(self.images))]

        recoverGroup = VGroup(TextMobject(
            "We can recover the original images"), TextMobject("with the help of eigen faces")).arrange(direction=DOWN)
        recoverGroupText2 = TextMobject(
            "Let's start with 50 principle components")
        self.play(Write(recoverGroup))
        self.wait(5)
        self.clear()
        self.play(Write(recoverGroupText2))
        self.wait(5)
        self.clear()

        recoveredResult = self.plot_portraits2(recovered_images, h, w, 4, 4)
        recoveredResult.move_to(2.6*LEFT)
        result2.move_to(2.6*RIGHT)
       # result3 = result2

        self.play(FadeIn(recoveredResult))
        self.wait(2)
        self.play(FadeIn(result2))
        self.wait(5)

        comparImage1 = result2[0]
        comparImage1.scale(2)
        comparImage1.move_to(2.3*RIGHT)
        groupy2 = Group()
        groupy2.add(comparImage1)
        compareImage2 = recoveredResult[0]
        compareImage2.scale(2)
        compareImage2.move_to(2.3*LEFT)
        groupy3 = Group()
        groupy3.add(compareImage2)

        self.play(Transform(result2, groupy2),
                  Transform(recoveredResult, groupy3))

        self.wait()

        brace1 = Brace(comparImage1, direction=TOP)
        braceText1 = brace1.get_text("Original Image")
        braceText1.scale(0.5)
        brace2 = Brace(compareImage2, direction=TOP)
        braceText2 = brace2.get_text("Recovered Image")
        braceText2.scale(0.5)
        self.play(GrowFromCenter(brace1), FadeIn(braceText1),
                  GrowFromCenter(brace2), FadeIn(braceText2))
        self.wait(5)

        textGroup = VGroup(TextMobject("The recovered image is not very clear"), TextMobject(
            "Let's increase no. of principle components to 300")
        ).arrange(direction=DOWN)

        textGroup.shift(DOWN*2)
        textGroup.scale(0.6)

        self.play(Write(textGroup))
        self.wait(5)
        self.clear()

        recoveredText2 = TextMobject("The Result")
        self.play(Write(recoveredText2))
        self.wait(3)
        self.clear()

        n_components = 300
        X = self.images.reshape(n_samples, h*w)
        P, C, M, Y = self.pca(X, n_components)

        eigenfaces = C.reshape((n_components, h, w))

        plt.figure(figsize=(2.2 * 4, 2.2 * 4))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(4 * 4):
            plt.subplot(4, 4, i + 1)
            plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
        plt.savefig("output.png")

        recovered_images2 = [self.reconstruction(
            Y, C, M, h, w, i) for i in range(len(self.images))]

        recoveredResult2 = self.plot_portraits2(recovered_images2, h, w, 4, 4)
        recoveredResult2.move_to(2.6*LEFT)
        result3.move_to(2.6*RIGHT)

        self.play(FadeIn(recoveredResult2))
        self.wait(2)
        self.play(FadeIn(result3))
        self.wait(5)

        comparImage3 = result3[0]
        comparImage3.scale(2)
        comparImage3.move_to(2.3*RIGHT)
        groupy4 = Group()
        groupy4.add(comparImage3)
        compareImage4 = recoveredResult2[0]
        compareImage4.scale(2)
        compareImage4.move_to(2.3*LEFT)
        groupy5 = Group()
        groupy5.add(compareImage4)

        self.play(Transform(result3, groupy4),
                  Transform(recoveredResult2, groupy5))

        self.wait()

        brace1 = Brace(comparImage3, direction=TOP)
        braceText1 = brace1.get_text("Original Image")
        braceText1.scale(0.5)
        brace2 = Brace(compareImage4, direction=TOP)
        braceText2 = brace2.get_text("Recovered Image")
        braceText2.scale(0.5)
        self.play(GrowFromCenter(brace1), FadeIn(braceText1),
                  GrowFromCenter(brace2), FadeIn(braceText2))
        self.wait(5)

        text2 = TextMobject(
            "The recovered and original images match perfectly!!")
        text2.scale(0.8)
        text2.shift(DOWN*2)

        self.play(Write(text2))
        self.wait(5)
        self.clear()

    def plot_portraits(self, images, h, w, n_row, n_col):
        plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
        bigGroup = Group()
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row * n_col):
            window = ImageMobject(
                images[i].reshape((h, w)))
            window.arrange(direction=DOWN)
            window.scale(0.6)
            bigGroup.add(window).arrange(direction=RIGHT)
        bigGroup.arrange_in_grid(4, 4)
        return bigGroup

    def plot_portraits2(self, images, h, w, n_row, n_col):
        plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
        bigGroup = Group()
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row * n_col):
            window = ImageMobject(
                images[i].reshape((h, w)))
            window.arrange(direction=DOWN)
            window.scale(0.4)
            bigGroup.add(window).arrange(direction=RIGHT)
        bigGroup.arrange_in_grid(4, 4)
        return bigGroup

    def pca(self, X, n_pc):
        n_samples, n_features = X.shape
        mean = np.mean(X, axis=0)
        centered_data = X-mean
        U, S, V = np.linalg.svd(centered_data)
        components = V[:n_pc]
        projected = U[:, :n_pc]*S[:n_pc]
        return projected, components, mean, centered_data

    def reconstruction(self, Y, C, M, h, w, image_index):
        weights = np.dot(Y, C.T)
        centered_vector = np.dot(weights[image_index, :], C)
        recovered_image = (M+centered_vector).reshape(h, w)
        return recovered_image

    def convertImage(self, image):
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
