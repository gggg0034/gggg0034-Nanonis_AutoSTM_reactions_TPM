"""nanonis_guard.py -- friendly diagnostics for the Nanonis TCP link.

The instrument API in ``interface.py`` opens a TCP socket to the Nanonis
software in its constructor.  When Nanonis is not running -- which is the
normal situation on a reviewer's laptop -- the raw failure is a
``ConnectionRefusedError`` raised from deep inside the import chain, and the
traceback says nothing about what the user is supposed to do.

This module turns that into a short, actionable message, and provides the
``nanonis_available()`` probe used by the entry points to decide between the
recorded-data demo and a real experiment.
"""

import socket

NANONIS_DEFAULT_IP = "127.0.0.1"
NANONIS_DEFAULT_PORT = 6501


class NanonisConnectionError(RuntimeError):
    """Raised when the TCP link to Nanonis cannot be established."""


def build_message(ip=NANONIS_DEFAULT_IP, port=NANONIS_DEFAULT_PORT, detail=""):
    lines = [
        "",
        "=" * 78,
        "  Cannot connect to the Nanonis controller.",
        "  无法连接到 Nanonis 控制器。",
        "-" * 78,
        "  What this means",
        "    The program talks to Nanonis through a TCP socket on %s:%d." % (ip, port),
        "    Nothing answered on that port, so the instrument cannot be driven.",
        "",
        "  What to check",
        "    1. Nanonis software is running on this computer (or on the computer",
        "       whose IP you passed in).",
        "    2. A Nanonis license is installed AND that license includes the",
        "       TCP/IP programming interface. Without it the server socket never",
        "       opens, so the port stays closed even though Nanonis is running.",
        "    3. In Nanonis:  File > Preferences > TCP Programming Interface,",
        "       make sure the server is enabled and the port number matches %d." % port,
        "    4. No other program is already holding that port. Nanonis accepts a",
        "       single client per port; if 6501 is busy, try 6502, 6503 or 6504.",
        "",
        "  What you can still do without an instrument",
        "    python demo_pipeline.py      # recorded scan + keypoint + decision demo",
        "    python train_sim.py          # train the SAC agent in the simulator",
        "    python run_bo.py             # run the Bayesian-optimisation baseline",
        "-" * 78,
    ]
    if detail:
        lines.append("  Underlying error: %s" % detail)
        lines.append("-" * 78)
    lines.append("=" * 78)
    lines.append("")
    return "\n".join(lines)


def nanonis_available(ip=NANONIS_DEFAULT_IP, port=NANONIS_DEFAULT_PORT, timeout=1.0):
    """Return True if something accepts a TCP connection on ip:port.

    Only used to decide which mode to start in; it does not prove that the
    Nanonis programming interface is licensed, only that the socket is open.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def require_nanonis(ip=NANONIS_DEFAULT_IP, port=NANONIS_DEFAULT_PORT):
    """Raise NanonisConnectionError with the message above if the port is closed."""
    if not nanonis_available(ip, port):
        raise NanonisConnectionError(build_message(ip, port))
