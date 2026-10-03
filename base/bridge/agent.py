"""An agent in front of a server that already speaks the click protocol of
nnInteractive (SLIP's server does, and nnInteractive's own): the kit takes
care of the platform's side (token, manifest, uploads in pieces, sessions),
and each call is passed on to that server, which runs beside it in the same
container and is reached by nobody else.

Such a server holds one image at a time, so the manifest of an agent built
this way says `maxSessions: 1`: the platform gives it to one user at a time.

Environment: BRIDGE_COMMAND (how the server is started, e.g. "uvicorn app:app
--host 127.0.0.1 --port 1529"; left out when something else starts it),
BRIDGE_URL (where it listens, default http://127.0.0.1:1529), BRIDGE_START_S
(how long it may take to come up, default 600).
"""
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from seg_agent_kit import ImageRefused, Mask, Volume, serve

URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:1529")
START_S = float(os.environ.get("BRIDGE_START_S", "600"))
# The platform gives up on a call after the time the manifest allows; this only ends a call that never would.
CALL_S = 3600


def _post(path, body, content_type):
    """One call to the server: the body of its answer. A refusal is said in the server's own words."""
    request = urllib.request.Request(URL + path, data=body, headers={"content-type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=CALL_S) as answer:
            return answer.read()
    except urllib.error.HTTPError as e:
        try:
            said = json.loads(e.read()).get("error")
        except (ValueError, AttributeError):
            said = None
        raise RuntimeError(said or f"the model server answered HTTP {e.code}") from None


def _form(fields):
    """A multipart form, as (body, content type), of fields given as (name,
    content, file name), the file name being None for a plain value."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, content, filename in fields:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\ncontent-disposition: {disposition}\r\n\r\n".encode() + content + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def _array(shape, dtype, data):
    """An array as the form field `file` the server reads it from."""
    return ("file", Volume(shape, dtype, data).npy(), "array.npy")


def _mask(answer, shape):
    """The mask the server answered a prompt with, which must be of the image's shape."""
    mask = Volume.from_npy(answer)
    if mask.shape != tuple(shape):
        raise RuntimeError("the model server answered a mask of another shape")
    return Mask(mask.shape, bytearray(mask.data))


class Session:
    """One image on the server, and the prompts made on it."""

    def __init__(self, image):
        self.shape = image.shape
        try:
            _post("/upload_image", *_form([_array(image.shape, image.dtype, image.data)]))
        except RuntimeError as e:
            raise ImageRefused(str(e)) from None

    def point(self, at, positive):
        body = json.dumps({"voxel_coord": list(at), "positive_click": positive}).encode()
        return _mask(_post("/add_point_interaction", body, "application/json"), self.shape)

    def bbox(self, one, two, positive):
        body = json.dumps({"outer_point_one": list(one), "outer_point_two": list(two), "positive_click": positive}).encode()
        return _mask(_post("/add_bbox_interaction", body, "application/json"), self.shape)

    def scribble(self, mask, positive):
        return self._drawn("/add_scribble_interaction", mask, positive)

    def lasso(self, mask, positive):
        return self._drawn("/add_lasso_interaction", mask, positive)

    def reset(self, seed):
        """Start a new object from `seed` (a mask, empty for none)."""
        _post("/upload_segment", *_form([_array(seed.shape, "|u1", seed.data)]))

    def _drawn(self, path, mask, positive):
        fields = [_array(mask.shape, "|u1", mask.data), ("positive_click", b"true" if positive else b"false", None)]
        return _mask(_post(path, *_form(fields)), self.shape)


def load():
    """Starts the server when told how, and waits until it answers."""
    command = os.environ.get("BRIDGE_COMMAND")
    server = subprocess.Popen(shlex.split(command)) if command else None
    deadline = time.monotonic() + START_S
    while time.monotonic() < deadline:
        if server and server.poll() is not None:
            raise RuntimeError(f"the model server stopped with code {server.returncode}")
        try:
            with urllib.request.urlopen(URL + "/", timeout=5):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("the model server did not come up in time")


if __name__ == "__main__":
    serve(Session, load=load)
