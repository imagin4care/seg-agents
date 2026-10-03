"""An example agent, to replace with yours: click in a structure, and it grows
a region from the voxel you clicked through its neighbours of about the same
intensity. No model and no GPU; it shows where your model's code goes.

The kit beside this file (seg_agent_kit.py) does the rest: the platform's
token, the uploads, one Session per user and image, the encoding of masks.
"""
from collections import deque

from seg_agent_kit import Mask, serve

# How far a voxel may be from the clicked one, as a share of the image's range.
TOLERANCE = 0.08
# A region stops growing here: this is plain Python, and a click must answer in seconds.
MAX_VOXELS = 150_000
NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class Session:
    """One user's work on one image. The kit makes one for each image sent."""

    def __init__(self, image):
        self.shape = image.shape  # (z, y, x)
        self.values = image.values()  # every voxel: index = (z * ny + y) * nx + x. With NumPy: image.numpy()
        self.span = (max(self.values) - min(self.values)) or 1
        self.mask = Mask.empty(image.shape)

    def reset(self, seed):
        """A new object is started, from `seed`: a mask, empty for none."""
        self.mask = seed

    def point(self, at, positive):
        """A click at `at` (z, y, x): in the object when `positive`, outside
        it when not. Returns the whole object so far."""
        nz, ny, nx = self.shape
        if not all(0 <= c < n for c, n in zip(at, self.shape)):
            return self.mask
        start = (at[0] * ny + at[1]) * nx + at[2]
        wanted, slack = self.values[start], TOLERANCE * self.span
        seen, queue = {start}, deque([start])
        while queue and len(seen) <= MAX_VOXELS:
            i = queue.popleft()
            self.mask.data[i] = 1 if positive else 0
            z, rest = divmod(i, ny * nx)
            y, x = divmod(rest, nx)
            for dz, dy, dx in NEIGHBOURS:
                zz, yy, xx = z + dz, y + dy, x + dx
                if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                    j = (zz * ny + yy) * nx + xx
                    if j not in seen and abs(self.values[j] - wanted) <= slack:
                        seen.add(j)
                        queue.append(j)
        return self.mask


if __name__ == "__main__":
    # serve(Session, load=read_the_model) calls `read_the_model` once before the agent says it is ready.
    serve(Session)
