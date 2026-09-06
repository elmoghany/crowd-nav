"""Watch a running simulation from another machine, in a browser.

The simulation renders on whatever box it runs on (a Linux server, a cluster node) and serves
the frames as MJPEG over plain HTTP -- one TCP port, no plugins, no UDP, so it survives an SSH
tunnel as happily as a direct connection.

    python -m sim.simulate --scenario doorway --serve 8080

then open http://<that machine>:8080 from your laptop. See REMOTE.md.
"""
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><meta charset="utf-8"><title>crowd-nav live</title>
<style>
 body{margin:0;background:#12141a;color:#e8eaee;font:14px system-ui;display:grid;
      place-items:center;min-height:100vh}
 img{max-width:96vw;border-radius:6px;border:1px solid #2b303a}
 p{opacity:.65;font:12px ui-monospace,Menlo,Consolas,monospace;margin:.8rem}
</style>
<img src="/stream" alt="live simulation">
<p>crowd-nav &mdash; live MJPEG. The run ends on its own; reload if the image freezes.</p>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "crowd-nav-sim"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last = -1
        try:
            while not self.server.stop_flag.is_set():
                with self.server.cond:
                    self.server.cond.wait_for(
                        lambda: self.server.seq != last or self.server.stop_flag.is_set(), timeout=5)
                    if self.server.jpeg is None:
                        continue
                    buf, last = self.server.jpeg, self.server.seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n")
                self.wfile.write(buf)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                      # viewer closed the tab; nothing to clean up

    def log_message(self, *a):        # keep the simulation's own output readable
        pass


class FrameServer:
    """Serves the latest rendered frame as MJPEG. `publish()` is called from the sim loop."""

    def __init__(self, port, host="0.0.0.0", quality=80):
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.jpeg, self.httpd.seq = None, 0
        self.httpd.cond = threading.Condition()
        self.httpd.stop_flag = threading.Event()
        self.quality = quality
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def urls(self):
        """Every address this is reachable on, so the user can copy one."""
        out = ["http://localhost:%d" % self.port]
        try:
            host = socket.gethostname()
            out.append("http://%s:%d" % (host, self.port))
            ip = socket.gethostbyname(host)
            if not ip.startswith("127."):
                out.append("http://%s:%d" % (ip, self.port))
        except OSError:
            pass
        return out

    def publish(self, rgb):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=self.quality)
        with self.httpd.cond:
            self.httpd.jpeg = buf.getvalue()
            self.httpd.seq += 1
            self.httpd.cond.notify_all()

    def close(self):
        self.httpd.stop_flag.set()
        with self.httpd.cond:
            self.httpd.cond.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
