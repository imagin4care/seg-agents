"""Where an image is in space, as the platform says it and as NIfTI wants it;
shared by the base agents that hand the image to a model reading NIfTI.

The platform gives the spacing of the voxels, the position of voxel (0, 0, 0)
and the direction of the three axes (a 3×3 matrix, row by row) in the
patient space of DICOM, LPS. NIfTI's is RAS: the first two rows change sign.
Standard library only, so that it is tested without a model.
"""

# No body is longer: an image that says it is has a voxel spacing that cannot
# be right, and resampling it would take more memory than a worker has.
MAX_EXTENT_MM = 2500


def extent_mm(shape, geometry):
    """How far the image reaches along each of its axes, in millimetres. The
    voxels are counted (z, y, x), the spacing is given (x, y, z)."""
    return [n * s for n, s in zip(shape, reversed(geometry["spacing"]))]


def ras_affine(geometry):
    """The 4×4 NIfTI affine (rows of four numbers) of an image at `geometry`."""
    direction, spacing, origin = geometry["direction"], geometry["spacing"], geometry["origin"]
    rows = [[direction[r * 3 + c] * spacing[c] for c in range(3)] + [origin[r]] for r in range(3)]
    return [[-v for v in rows[0]], [-v for v in rows[1]], rows[2], [0.0, 0.0, 0.0, 1.0]]


def refusal(shape, geometry):
    """Why an image of `shape` at `geometry` cannot be segmented as a body,
    in words; None when it can."""
    if not geometry:
        return "where the image is in space is needed: its spacing, origin and direction"
    if max(extent_mm(shape, geometry)) > MAX_EXTENT_MM:
        return "it says it is larger than a body: its voxel spacing cannot be right"
    return None
