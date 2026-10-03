"""seg_agent_kit — the plumbing of an agent of the segmentation marketplace.

An author writes what the agent does; this file serves it the way the
platform calls it. An interactive agent (contract "session", version 1):

    from seg_agent_kit import Mask, serve

    class Session:                       # one instance per user and image
        def __init__(self, image):       # image: Volume (shape z, y, x)
            self.mask = Mask.empty(image.shape)
        def point(self, at, positive):   # at: (z, y, x)
            ...
            return self.mask             # the whole object so far

    serve(Session)

An automatic agent (contract "job", version 1):

    from seg_agent_kit import LabelMap, serve

    def segment(image, job):             # job.geometry, job.parameters
        labels = LabelMap.empty(image.shape)
        ...
        job.progress(0.5, "half way")    # and job.cancelled() to stop early
        return labels                    # the values its manifest declares

    serve(job=segment)

An agent that answers in words (contract "chat", version 1):

    from seg_agent_kit import serve

    def interpret(pictures, question, context):   # pictures: Picture (media_type, data, caption)
        ...
        return "What the slices show, in words."

    serve(interpret=interpret)

What it takes care of: the token check, `/ping`, `/manifest`, uploads in
pieces (`/_relay`), one session per `x-agent-session`, jobs run in the
background and asked about meanwhile, the limits on what a caller may send,
and the encoding of volumes, masks and label maps. Standard library only;
`numpy()` and `from_numpy()` are there for those who have NumPy.

Environment: PORT (default 8000), AGENT_TOKEN (set by the platform; without
it the agent serves nobody), AGENT_MANIFEST (path, default manifest.json).
"""
import array
import ast
import base64
import gzip
import hmac
import json
import os
import re
import secrets
import struct
import sys
import threading
import time
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["Volume", "Mask", "LabelMap", "ImageRefused", "serve"]

_MAGIC = b"\x93NUMPY"
_ITEM_SIZE = re.compile(r"^[<>|=]?[a-zA-Z](\d+)$")
_PART_NAME = re.compile(rb'(?:^|[;\s])name="([^"]*)"')
_RELAY_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
# The largest volume: what an upload in pieces may add up to, and what a
# compressed array may inflate to.
_MAX_VOLUME_BYTES = 2 * 1024 ** 3
# RunPod's load balancer carries 30 MB a request; the platform sends less.
_MAX_REQUEST_BYTES = 32 * 1024 ** 2
# An upload in pieces whose next piece does not come is forgotten after this.
_RELAY_IDLE_S = 600
# Room for the header of a `.npy` beside its values.
_HEADER_BYTES = 4096
# What a job says of itself is one short line: the platform reads no long answer.
_MESSAGE_CHARS = 200
# An answer in words: the platform reads no longer one.
_ANSWER_CHARS = 20_000
_PROMPTS = {
    "/add_point_interaction": "point",
    "/add_bbox_interaction": "bbox",
    "/add_scribble_interaction": "scribble",
    "/add_lasso_interaction": "lasso",
}


# What `Volume.values()` reads each type of array as.
_TYPECODES = {"|u1": "B", "|i1": "b", "<u2": "H", "<i2": "h", "<f4": "f", "<f8": "d"}


class ImageRefused(Exception):
    """Raise it from a session's constructor when the image cannot be used.
    The message is shown to the user."""


def _count(shape):
    count = 1
    for n in shape:
        count *= n
    return count


def _inflated(raw, limit):
    """`raw` gunzipped, refused past `limit` bytes: a few kilobytes of gzip
    can hold gigabytes."""
    inflater = zlib.decompressobj(wbits=31)
    data = inflater.decompress(raw, limit + 1)
    if len(data) > limit:
        raise ValueError("the array is too large")
    if not inflater.eof:
        raise ValueError("the array is cut short")
    return data


class Volume:
    """An array as it arrived: `shape` (z, y, x), `dtype` as NumPy writes it
    ("<i2", "|u1", ...) and `data`, the values in C order."""

    def __init__(self, shape, dtype, data):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.data = data

    def numpy(self):
        import numpy
        return numpy.frombuffer(self.data, dtype=numpy.dtype(self.dtype)).reshape(self.shape)

    def npy(self):
        """The array as a `.npy` file, to hand it to another program."""
        return _npy(self.dtype, self.shape, bytes(self.data))

    def values(self):
        """The values as numbers, in C order, for those without NumPy."""
        code = _TYPECODES.get(self.dtype)
        if code is None:
            raise ValueError(f"arrays of type {self.dtype} cannot be read as numbers here")
        values = array.array(code)
        values.frombytes(bytes(self.data))
        if sys.byteorder == "big":
            values.byteswap()
        return values

    @staticmethod
    def from_npy(raw, limit=_MAX_VOLUME_BYTES):
        """Reads a `.npy` file, gzipped or not, of `limit` bytes at most."""
        if raw[:2] == b"\x1f\x8b":
            raw = _inflated(raw, limit)
        if raw[:6] != _MAGIC:
            raise ValueError("not a .npy file")
        if raw[6] == 1:
            (length,), start = struct.unpack("<H", raw[8:10]), 10
        else:
            (length,), start = struct.unpack("<I", raw[8:12]), 12
        header = ast.literal_eval(raw[start:start + length].decode("latin1"))
        if header.get("fortran_order"):
            raise ValueError("Fortran-ordered arrays are not accepted")
        shape, dtype = tuple(header["shape"]), header["descr"]
        item = _ITEM_SIZE.match(dtype)
        if not item:
            raise ValueError(f"arrays of type {dtype} are not accepted")
        data = raw[start + length:]
        if len(data) != _count(shape) * int(item.group(1)):
            raise ValueError("the array is cut short")
        return Volume(shape, dtype, data)


class Mask:
    """A mask of an image: one byte per voxel, 0 or 1, in C order (z, y, x)."""

    def __init__(self, shape, data):
        self.shape = tuple(shape)
        self.data = data

    @staticmethod
    def empty(shape):
        return Mask(shape, bytearray(_count(shape)))

    @staticmethod
    def from_numpy(array):
        return Mask(array.shape, (array != 0).astype("uint8").tobytes())

    def numpy(self):
        import numpy
        return numpy.frombuffer(self.data, dtype="uint8").reshape(self.shape)

    def set(self, z, y, x, value=1):
        nz, ny, nx = self.shape
        if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
            self.data[(z * ny + y) * nx + x] = value

    def encoded(self):
        """The mask as the platform expects it: a gzipped `.npy` of uint8."""
        return _npy_gz("|u1", self.shape, bytes(self.data))


class LabelMap:
    """A label map of an image: one value per voxel, 0 where there is
    nothing, in C order (z, y, x). `bits` is 8, or 16 for values over 255."""

    def __init__(self, shape, data, bits=8):
        self.shape = tuple(shape)
        self.data = data
        self.bits = bits

    @staticmethod
    def empty(shape, bits=8):
        return LabelMap(shape, array.array("B" if bits == 8 else "H", bytes(_count(shape) * bits // 8)), bits)

    @staticmethod
    def from_numpy(labels):
        bits = 8 if labels.max(initial=0) <= 255 else 16
        return LabelMap(labels.shape, array.array("B" if bits == 8 else "H", labels.astype(f"uint{bits}").tobytes()), bits)

    def set(self, z, y, x, value):
        nz, ny, nx = self.shape
        if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
            self.data[(z * ny + y) * nx + x] = value

    def encoded(self):
        """The label map as the platform expects it: a gzipped `.npy` of uint8 or uint16."""
        values = self.data
        if self.bits == 16 and sys.byteorder == "big":
            values = array.array("H", values)
            values.byteswap()
        return _npy_gz("|u1" if self.bits == 8 else "<u2", self.shape, values.tobytes())


def _npy(descr, shape, payload):
    """A `.npy` file (format 1.0) of `payload`, the values of an array of `shape` and type `descr`."""
    dims = ", ".join(str(n) for n in shape) + ("," if len(shape) == 1 else "")
    header = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (descr, dims)
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    return _MAGIC + b"\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + payload


def _npy_gz(descr, shape, payload):
    """The same, gzipped: what the platform is answered with."""
    return gzip.compress(_npy(descr, shape, payload), compresslevel=1)


def _multipart(content_type, body):
    """The parts of a multipart form: name -> content."""
    match = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type or "")
    if not match:
        raise ValueError("not a multipart form")
    delimiter = b"--" + (match.group(1) or match.group(2)).encode("latin1")
    parts = {}
    for chunk in body.split(delimiter)[1:]:
        if chunk[:2] == b"--":
            break
        head, _, content = chunk.partition(b"\r\n\r\n")
        name = _PART_NAME.search(head)
        if name:
            parts[name.group(1).decode("latin1")] = content[:-2] if content.endswith(b"\r\n") else content
    return parts


class _Answer(Exception):
    """An answer that is not a result: a status, and why."""

    def __init__(self, status, error, message=""):
        super().__init__(error)
        self.status = status
        self.body = json.dumps({"error": error, "message": message}).encode("utf8")


class _Entry:
    """One session of the worker: the author's object, the shape of its
    image, and the lock that keeps its calls one at a time."""

    def __init__(self, session, shape):
        self.session = session
        self.shape = shape
        self.lock = threading.Lock()

    def drawn(self, content_type, body):
        """The mask a form carries in `file`, of the image's shape, and the
        other fields of the form."""
        parts = _multipart(content_type, body)
        mask = Volume.from_npy(parts["file"], limit=_count(self.shape) + _HEADER_BYTES)
        if mask.shape != self.shape:
            raise ValueError("the mask is not of the shape of the image")
        return Mask(mask.shape, bytearray(mask.data)), parts

    def close(self):
        if hasattr(self.session, "close"):
            self.session.close()


class _Sessions:
    """The sessions of one worker, and the calls of the session contract on them."""

    def __init__(self, session_class, max_sessions):
        self.session_class = session_class
        self.max_sessions = max_sessions
        self.sessions = {}  # session id -> _Entry, the one used longest ago first
        self.lock = threading.Lock()

    def call(self, method, path, session_id, content_type, body):
        """One call of the session contract. Returns (status, content type, body)."""
        if method == "DELETE" and path == "/session":
            with self.lock:
                entry = self.sessions.pop(session_id, None)
            if entry:
                entry.close()
            return 200, "application/json", b"{}"
        if method != "POST":
            raise _Answer(404, "not-found")
        if path == "/upload_image":
            return self._open(session_id, content_type, body)
        with self.lock:
            entry = self.sessions.pop(session_id, None)
            if entry:
                self.sessions[session_id] = entry  # used last
        if entry is None:
            raise _Answer(409, "no-session")
        with entry.lock:
            if path == "/upload_segment":
                if hasattr(entry.session, "reset"):
                    entry.session.reset(entry.drawn(content_type, body)[0])
                return 200, "application/json", b'{"status": "ok"}'
            kind = _PROMPTS.get(path)
            if kind is None or not hasattr(entry.session, kind):
                raise _Answer(404, "not-found")
            return 200, "application/octet-stream", self._prompt(entry, kind, content_type, body).encoded()

    def _open(self, session_id, content_type, body):
        try:
            image = Volume.from_npy(_multipart(content_type, body)["file"])
        except (KeyError, ValueError, AttributeError, SyntaxError) as e:
            raise _Answer(400, "bad-upload", str(e))
        try:
            entry = _Entry(self.session_class(image), image.shape)
        except ImageRefused as e:
            raise _Answer(422, "image-refused", str(e))
        with self.lock:
            dropped = [self.sessions.pop(session_id, None)]
            # The platform opens no more sessions than the manifest allows:
            # one too many means it closed a session and was not heard. The
            # session used longest ago is that one.
            while len(self.sessions) >= self.max_sessions:
                dropped.append(self.sessions.pop(next(iter(self.sessions))))
            self.sessions[session_id] = entry
        for old in dropped:
            if old:
                old.close()
        return 200, "application/json", b'{"status": "ok"}'

    @staticmethod
    def _prompt(entry, kind, content_type, body):
        if kind in ("point", "bbox"):
            prompt = json.loads(body)
            positive = prompt.get("positive_click", True)
            if kind == "point":
                return entry.session.point(tuple(prompt["voxel_coord"]), positive)
            return entry.session.bbox(tuple(prompt["outer_point_one"]), tuple(prompt["outer_point_two"]), positive)
        mask, parts = entry.drawn(content_type, body)
        positive = parts.get("positive_click", b"true").strip() != b"false"
        return getattr(entry.session, kind)(mask, positive)


class Job:
    """What a job handler is given beside the image: where the image is in
    space (`geometry`: its spacing, origin and direction), the `parameters`
    it was asked with, `progress(fraction, message)` to say how far it is
    (in one short line), and `cancelled()` to learn that nobody waits for the
    result any more. A handler that never looks at `cancelled()` runs to its
    end for nobody: look at it between two steps of the work."""

    def __init__(self, geometry, parameters):
        self.geometry = geometry
        self.parameters = parameters
        self.state = "queued"
        self.fraction = 0.0
        self.message = ""
        self.result = None
        self._cancel = threading.Event()

    def progress(self, fraction, message=""):
        self.fraction = max(0.0, min(1.0, float(fraction)))
        self.message = str(message)[:_MESSAGE_CHARS]

    def cancelled(self):
        return self._cancel.is_set()

    def over(self):
        """Nobody will ask for its result: it failed, or was told to stop."""
        return self.state in ("failed", "cancelled") or self.cancelled()


class _Jobs:
    """The jobs of one worker, and the calls of the job contract on them."""

    def __init__(self, run, max_jobs):
        self.run = run
        self.max_jobs = max_jobs
        self.jobs = {}  # job id -> Job, the oldest first
        self.lock = threading.Lock()

    def call(self, method, path, content_type, body, spec):
        """One call of the job contract. `spec`: what the job is asked with
        (the `x-agent-job` header). Returns (status, content type, body)."""
        parts = path.strip("/").split("/")[1:]
        if method == "POST" and not parts:
            return self._start(content_type, body, spec)
        with self.lock:
            job = self.jobs.get(parts[0]) if parts else None
        if job is None:
            raise _Answer(404, "no-job")
        if method == "GET" and len(parts) == 1:
            said = {"state": job.state, "progress": job.fraction, "message": job.message}
            return 200, "application/json", json.dumps(said).encode("utf8")
        if method == "GET" and parts[1:] == ["result"]:
            if job.state != "succeeded":
                raise _Answer(409, "not-done")
            # Given once: a label map is not kept for nobody.
            with self.lock:
                self.jobs.pop(parts[0], None)
            return 200, "application/octet-stream", job.result
        if method == "POST" and parts[1:] == ["cancel"]:
            job._cancel.set()
            return 200, "application/json", b"{}"
        raise _Answer(404, "not-found")

    def _start(self, content_type, body, spec):
        try:
            image = Volume.from_npy(_multipart(content_type, body)["file"])
            asked = json.loads(base64.urlsafe_b64decode(spec + "=" * (-len(spec) % 4))) if spec else {}
        except (KeyError, ValueError, AttributeError, SyntaxError) as e:
            raise _Answer(400, "bad-upload", str(e))
        job = Job(asked.get("geometry"), asked.get("parameters") or {})
        job_id = secrets.token_hex(8)
        with self.lock:
            # A job that is over gives its place first. With none of those,
            # one job too many means the platform gave one up and was not
            # heard, as for sessions: the oldest is that one.
            while len(self.jobs) >= self.max_jobs:
                leaving = next((i for i, j in self.jobs.items() if j.over()), next(iter(self.jobs)))
                self.jobs.pop(leaving)._cancel.set()
            self.jobs[job_id] = job
        threading.Thread(target=self._work, args=(job, image), daemon=True).start()
        return 202, "application/json", json.dumps({"id": job_id}).encode("utf8")

    def _work(self, job, image):
        job.state = "running"
        try:
            labels = self.run(image, job)
            if job.cancelled():
                job.state = "cancelled"
                return
            if not isinstance(labels, LabelMap) or labels.shape != image.shape:
                raise ValueError("the agent must return a LabelMap of the shape of the image")
            job.result = labels.encoded()
            job.fraction = 1.0
            job.state = "succeeded"
        except ImageRefused as e:
            job.message = str(e)[:_MESSAGE_CHARS]
            job.state = "failed"
        except Exception as e:  # noqa: BLE001 — whatever the author's code raises is the job's failure, said to the platform
            job.message = f"{type(e).__name__}: {e}"[:_MESSAGE_CHARS]
            job.state = "failed"


class Picture:
    """One image an agent is asked about: a rendered slice as a file
    (`data`, of type `media_type`: image/jpeg or image/png), and the line
    that says what it shows (`caption`)."""

    def __init__(self, media_type, data, caption):
        self.media_type = media_type
        self.data = data
        self.caption = caption


def _interpreted(interpret, body):
    """The one call of the chat contract: the question and its pictures in,
    the answer out. Returns (status, content type, body)."""
    try:
        asked = json.loads(body)
        pictures = [Picture(str(p["mediaType"]), base64.b64decode(p["data"], validate=True), str(p.get("caption", "")))
                    for p in asked.get("images", [])]
        question, context = str(asked["question"]), str(asked.get("context", ""))
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise _Answer(400, "bad-question", str(e))
    text = interpret(pictures, question, context)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the agent must answer a text")
    return 200, "application/json", json.dumps({"text": text[:_ANSWER_CHARS]}).encode("utf8")


class _Worker:
    """What one worker serves: its manifest, whether it is ready, and the
    contracts its author wrote."""

    def __init__(self, manifest, session_class, job, interpret):
        self.manifest = manifest
        self.ready = threading.Event()
        places = manifest.get("runtime", {}).get("maxSessions", 1)
        self.sessions = _Sessions(session_class, places) if session_class else None
        self.jobs = _Jobs(job, places) if job else None
        self.interpret = interpret

    def call(self, method, path, session_id, content_type, body, spec):
        """One call, whether it came whole or through the relay.
        Returns (status, content type, body)."""
        if not session_id:
            raise _Answer(400, "no-session-id")
        if path == "/interpret":
            if self.interpret is None or method != "POST":
                raise _Answer(404, "not-found")
            return _interpreted(self.interpret, body)
        if path == "/jobs" or path.startswith("/jobs/"):
            if self.jobs is None:
                raise _Answer(404, "not-found")
            return self.jobs.call(method, path, content_type, body, spec)
        if self.sessions is None:
            raise _Answer(404, "not-found")
        return self.sessions.call(method, path, session_id, content_type, body)


def _handler(agent, token):
    relays = {}  # upload id -> [when its last piece came, the pieces so far]
    relays_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 — the base class's signature
            del format, args  # one line per call would drown the agent's own output

        def _send(self, status, content_type, body):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            # Fail closed: an agent without a token serves nobody.
            given = self.headers.get("x-agent-token", "")
            return bool(token) and hmac.compare_digest(given.encode("utf8"), token.encode("utf8"))

        def _body(self):
            """The request's body, or None when it is refused unread: the
            caller is not the platform, or sends more than the platform does.
            The connection is then closed, what was not read with it."""
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = -1
            refusal = (None if self._authorized() else (401, b'{"error": "unauthorized"}'))
            if refusal is None and not 0 <= length <= _MAX_REQUEST_BYTES:
                refusal = (413, b'{"error": "request-too-large"}')
            if refusal is None:
                return self.rfile.read(length)
            self.close_connection = True
            self._send(refusal[0], "application/json", refusal[1])
            return None

        def _serve(self):
            path = self.path.split("?")[0]
            # The health check of RunPod's load balancer: no token, and says
            # nothing about any session.
            if path == "/ping" and self.command == "GET":
                return self._send(200, "text/plain", b"ok") if agent.ready.is_set() else self._send(204, "text/plain", b"")
            body = self._body()
            if body is None:
                return None
            if path == "/manifest" and self.command == "GET":
                return self._send(200, "application/json", json.dumps(agent.manifest).encode("utf8"))
            session_id = self.headers.get("x-agent-session")
            spec = self.headers.get("x-agent-job", "")
            content_type = self.headers.get("content-type", "")
            try:
                parts = path.strip("/").split("/")
                if parts[0] == "_relay":
                    if len(parts) != 3 or not _RELAY_ID.match(parts[1]):
                        raise _Answer(404, "not-found")
                    if self.command == "PUT" and parts[2].isdigit():
                        return self._relay_piece(parts[1], int(parts[2]), body)
                    if self.command == "POST" and parts[2] == "commit":
                        with relays_lock:
                            whole = b"".join(relays.pop(parts[1], (0, []))[1])
                        path = self.headers.get("x-relay-path", "")
                        content_type = self.headers.get("x-relay-content-type", "")
                        if not path.startswith("/") or path.startswith("/_relay"):
                            raise _Answer(400, "bad-relay-path")
                        return self._send(*agent.call("POST", path, session_id, content_type, whole, spec))
                    raise _Answer(404, "not-found")
                return self._send(*agent.call(self.command, path, session_id, content_type, body, spec))
            except _Answer as answer:
                return self._send(answer.status, "application/json", answer.body)
            except (KeyError, ValueError, TypeError) as e:
                return self._send(400, "application/json", json.dumps({"error": "bad-request", "message": str(e)}).encode("utf8"))
            except Exception as e:  # noqa: BLE001 — whatever else the author's code raises is answered, not left as a dropped connection
                traceback.print_exc()
                said = {"error": "agent-error", "message": f"{type(e).__name__}: {e}"[:_MESSAGE_CHARS]}
                return self._send(500, "application/json", json.dumps(said).encode("utf8"))

        def _relay_piece(self, upload_id, seq, body):
            """Pieces arrive in order from 0; a piece 0 starts the upload again."""
            now = time.monotonic()
            with relays_lock:
                # An upload nobody finishes does not stay in memory.
                for stale in [key for key, (at, _) in relays.items() if now - at > _RELAY_IDLE_S]:
                    del relays[stale]
                pieces = [] if seq == 0 else relays.get(upload_id, (0, None))[1]
                if pieces is None or seq != len(pieces):
                    raise _Answer(409, "piece-out-of-order")
                if sum(len(p) for p in pieces) + len(body) > _MAX_VOLUME_BYTES:
                    relays.pop(upload_id, None)
                    raise _Answer(413, "upload-too-large")
                relays[upload_id] = [now, pieces + [body]]
            return self._send(200, "application/json", b"{}")

        do_GET = do_POST = do_PUT = do_DELETE = _serve

    return Handler


def serve(session_class=None, job=None, interpret=None, load=None):
    """Serves an interactive agent (`session_class`), an automatic one (`job`,
    called with the image and a `Job`) or one that answers in words
    (`interpret`, called with the pictures, the question and the context)
    until the process is stopped.

    `load` (optional) is called once before the agent says it is ready: the
    place to read the model. While it runs, `/ping` answers 204. An agent
    whose `load` fails stops: a worker that can never be ready is not left
    starting.
    """
    with open(os.environ.get("AGENT_MANIFEST", "manifest.json"), encoding="utf8") as f:
        manifest = json.load(f)
    agent = _Worker(manifest, session_class, job, interpret)

    def become_ready():
        try:
            if load:
                load()
        except BaseException as e:  # noqa: BLE001 — whatever stops the model from loading, a sys.exit() included, stops the agent
            print(f"agent could not load: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        agent.ready.set()

    threading.Thread(target=become_ready, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler(agent, os.environ.get("AGENT_TOKEN", "")))
    server.daemon_threads = True
    print(f"agent {manifest.get('name')}@{manifest.get('version')} on :{port}", flush=True)
    server.serve_forever()
