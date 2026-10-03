"""The colour of each label of a base agent's manifest. Standard library only."""
import colorsys


def colour(value):
    """A colour per label value, far from its neighbours': the hue turns by the golden angle."""
    r, g, b = colorsys.hsv_to_rgb((value * 0.61803398875) % 1.0, 0.55 + 0.35 * ((value * 7) % 3) / 2, 0.95)
    return [round(r * 255), round(g * 255), round(b * 255)]
