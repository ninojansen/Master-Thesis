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
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    TEAL = (0, 128, 128)
    BROWN = (165, 42, 42)
    # Extension colors
    LIGHT_GRAY = (83, 83, 83)
    DARK_RED = (55, 0, 0)
    CYAN = (0, 100, 100)
    PURPLE = (50, 0, 50)
    PINK = (100, 75, 80)
    BEIGE = (96, 96, 86)
    ORANGE = (100, 65, 0)
    LIGHT_BLUE = (68, 85, 90)


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
