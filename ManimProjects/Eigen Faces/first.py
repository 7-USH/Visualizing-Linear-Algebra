from manimlib.imports import *
from manim import *


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        self.play(Write(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
