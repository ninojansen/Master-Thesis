from enum import Enum


class Shape(Enum):
    RECTANGLE = 1
    CIRCLE = 2
    TRIANGLE = 3
   # SQUARE = 4


class Size(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


class Color(Enum):
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    BROWN = (165, 42, 42)
    # Rainbow
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 128, 0)
    BLUE = (0, 0, 255)
    INDIGO = (75, 0, 130)
    VIOLET = (238, 130, 238)


class Location(Enum):
    # x_min, x_max, y_min, y_max %
    TOP_LEFT = (0, 0.5, 0, 0.5)
    TOP = (0, 1, 0, 0.5)
    TOP_RIGHT = (0.5, 1, 0, 0.5)
    LEFT = (0, 0.5, 0, 1)
    CENTRE = (0.25, 0.75, 0.25, 0.75)
    RIGHT = (0.5, 1, 0, 1)
    BOTTOM_LEFT = (0, 0.5, 0.5, 1)
    BOTTOM = (0, 1, 0.5, 1)
    BOTTOM_RIGHT = (0.5, 1, 0.5, 1)
