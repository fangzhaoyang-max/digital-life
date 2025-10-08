"""
Utility functions for the Digital Life system
"""

import socket
import os
import time
import hashlib
import json
import ipaddress
from typing import Any

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi"""
    return max(lo, min(hi, x))


def generate_node_id() -> str:
    """Generate a unique node ID"""
    host_info = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
    return hashlib.sha3_256(host_info.encode()).hexdigest()[:32]


def is_port_free(port: int, host: str = '0.0.0.0') -> bool:
    """
    Check if a port is free on the given host
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except Exception:
                pass
        s.bind((host, port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def find_free_port(start: int, end: int) -> int:
    """Find a free port in the given range"""
    for p in range(start, end + 1):
        if is_port_free(p, '0.0.0.0'):
            return p
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('0.0.0.0', 0))
        p = s.getsockname()[1]
        s.close()
        return p
    except Exception:
        return start


def auto_detect_host(current_host: str) -> str:
    """Auto-detect external IP address if using localhost"""
    try:
        if current_host in ('127.0.0.1', '0.0.0.0'):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.3)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip:
                return ip
    except Exception:
        pass
    return current_host


def json_sanitize(obj: Any, max_depth: int = 4) -> Any:
    """Convert object to JSON-serializable form with depth limit"""
    if max_depth <= 0:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(x, max_depth - 1) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            out[key] = json_sanitize(v, max_depth - 1)
        return out
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def is_public_destination(host: str) -> bool:
    """Check if host resolves to public IP addresses only"""
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
        addrs = {info[4][0] for info in infos}
        if not addrs:
            return False
        for ip in addrs:
            ip_obj = ipaddress.ip_address(ip)
            if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
                    or ip_obj.is_reserved or ip_obj.is_multicast):
                return False
        return True
    except Exception:
        return False


def is_local_address(host: str) -> bool:
    """Check if host is a local/private address"""
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
        addrs = {info[4][0] for info in infos}
        if not addrs:
            return False
        for ip in addrs:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_link_local:
                return True
        return False
    except Exception:
        return False


def is_allowed_destination(host: str, strict_check: bool = True, allow_local: bool = True) -> bool:
    """
    Check if a destination host is allowed for communication
    """
    if not strict_check:
        return True
    if is_public_destination(host):
        return True
    if allow_local and is_local_address(host):
        return True
    return False