from math import sqrt, floor
from abc import ABCMeta, abstractmethod
import numpy as np


class Shape(metaclass=ABCMeta):
    '''a shape representitive class.
        expected to be overwritten by its childern.
    '''

    def __init__(self, radius=10, box_size=20):
        '''pretended that a user send an array of radiuses'''
        self.box_size = box_size  # the entier image (box) size. the box is square.

        if isinstance(radius, int):
            radius = np.array([radius])

        elif isinstance(radius, float):
            radius = np.array([floor(radius)])

        elif not isinstance(radius, np.ndarray):
            self.shape_rad = np.array(radius)

        self.shape_rad = np.array(radius)
        self.shape_position = np.zeros(
            (self.shape_rad.size, box_size//2 * 2, box_size//2 * 2))

    @abstractmethod
    def get_image(self):
        '''gives back the shape on the canvas (these together are known an image).
            it is expected to return a 2d-matrix where 1s represent shapes and
            0s represent the white spots on canvas.
        '''
        pass


class Circle(Shape):
    def __init__(self, radius_size=10, box_size=20):
        super().__init__(radius_size, box_size)

    def get_image(self):
        '''implementing its parent (Shape class abstract method)'''

        for i, r in enumerate(self.shape_rad-1):
            for x in range(-self.box_size//2, self.box_size//2):
                for y in range(-self.box_size//2, self.box_size//2):
                    if (x**2+y**2 - r**2 < 0.1):
                        self.shape_position[i, x+self.box_size //
                                            2, y+self.box_size//2] = 1
        return self.shape_position


class Square(Shape):
    def __init__(self, square_size=10, box_size=20):
        super().__init__(square_size, box_size)

    def get_image(self):
        '''implementing its parent (Shape class abstract method)'''

        for i, r in enumerate(self.shape_rad-1):
            for x in range(-self.box_size//2, self.box_size//2):
                for y in range(-self.box_size//2, self.box_size//2):
                    if ((abs(x) - r) < 0.01 and (abs(y) - r) < 0.01):
                        self.shape_position[i, x+self.box_size //
                                            2, y+self.box_size//2] = 1

        return self.shape_position


class Line(Shape):
    def __init__(self, line_position=10, box_size=20, line_ori='x'):
        super().__init__(line_position, box_size)
        self.line_ori = line_ori

    def get_image(self):
        '''implementing its parent (Shape class abstract method)'''

        # (box_size//2) * 2
        for i, r in enumerate(self.shape_rad-1):
            for x in range(self.box_size//2 * 2):
                for y in range(self.box_size//2 * 2):

                    if (x >= y-r and (x <= y+r) and self.line_ori == 'diag'):
                        self.shape_position[i, x, y] = 1

                    if (self.box_size-x >= y-r and (self.box_size-x <= y+r) and self.line_ori == 'anti_diag'):
                        self.shape_position[i, x, y] = 1

                    elif (x >= r and (x <= r+1) and self.line_ori == 'x'):
                        self.shape_position[i, x, y] = 1

                    elif (y >= r and (y <= r+1) and self.line_ori == 'y'):
                        self.shape_position[i, x, y] = 1

        return self.shape_position
