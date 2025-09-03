#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import ssl
import ast
import zlib
import json
import time
import uuid
import math
import base64
import psutil
import queue
import types
import signal
import random
import string
import socket
import pickle
import logging
import inspect
import textwrap
import threading
import tempfile
import datetime
from dataclasses import dataclass
from collections import deque, defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Deque, Callable, Union

import numpy as np
import requests
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flask import Flask, jsonify, request, make_response, g
from flask_cors import CORS

try:
    # Flask-Limiter >=3.0
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except Exception:
    Limiter = None

from pydantic import BaseModel, Field, ValidationError, validator

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import rsa, x25519, ed25519
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption
)

from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===========================
# 结构化日志
# ===========================

class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = record.trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

logger = logging.getLogger("TrueDigitalLife")
logger.setLevel(logging.INFO)
_stream = logging.StreamHandler()
_stream.setFormatter(JSONFormatter())
logger.addHandler(_stream)

_log_file = logging.handlers.RotatingFileHandler(
    "digital_life_rotating.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_log_file.setFormatter(JSONFormatter())
logger.addHandler(_log_file)

# ===========================
# Prometheus 指标
# ===========================
METRIC_REQUESTS = Counter("tdl_http_requests_total", "HTTP requests", ["endpoint", "method", "status"])
METRIC_HANDSHAKE_FAIL = Counter("tdl_handshake_fail_total", "Key exchange failures")
METRIC_DECRYPT_FAIL = Counter("tdl_decrypt_fail_total", "Decrypt failures")
METRIC_MESSAGES_SENT = Counter("tdl_messages_sent_total", "Messages sent", ["endpoint"])
METRIC_MESSAGES_RECV = Counter("tdl_messages_recv_total", "Messages received", ["endpoint"])
METRIC_NETWORK_ERRORS = Counter("tdl_network_errors_total", "Network errors")
METRIC_ACTION_DECISIONS = Counter("tdl_action_decisions_total", "Decisions", ["action"])
METRIC_ENERGY = Gauge("tdl_energy", "Current energy")
METRIC_THREAT = Gauge("tdl_threat_level", "Current threat level")
METRIC_STRESS = Gauge("tdl_stress", "Current stress")
METRIC_TRAIN_STEPS = Counter("tdl_training_steps_total", "NN training steps")
METRIC_EVOLUTION_ATTEMPTS = Counter("tdl_evolution_attempts_total", "Evolution attempts")
METRIC_EVOLUTION_ACCEPTED = Counter("tdl_evolution_accepted_total", "Evolution accepted")

# ===========================
# 安全配置
# ===========================
SECURITY_CONFIG = {
    'min_tls_version': ssl.TLSVersion.TLSv1_2,
    'cipher_list': 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384',
    'key_size': 4096,
    'cert_expiry_days': 365,
    'max_request_size': 10 * 1024 * 1024,  # 10MB
    'rate_limit': "100/minute",
    'api_timeout': 30,  # seconds
    'max_connections': 100,
    'heartbeat_interval': 60,  # seconds
    'data_retention_days': 7,
    'encryption_algorithm': 'Fernet(X25519+HKDF)',
    'signature_algorithm': 'Ed25519'
}

class NodeRole(Enum):
    QUEEN = auto()
    WORKER = auto()
    DRONE = auto()
    SENTINEL = auto()

class NetworkProtocol(Enum):
    HTTP = auto()
    HTTPS = auto()
    WS = auto()
    WSS = auto()
    TCP = auto()
    UDP = auto()
    GRPC = auto()

class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    PARANOID = auto()

class LifeState(Enum):
    ACTIVE = auto()
    SLEEP = auto()
    TERMINATED = auto()

# ===========================
# 配置模型（校验）
# ===========================
try:
    from pydantic.v1 import BaseModel as PydBaseModel
except Exception:
    PydBaseModel = BaseModel

class ConfigModel(PydBaseModel):
    energy_threshold: float = 30.0
    replication_threshold: float = 0.75
    mutation_rate: float = 0.03
    learning_rate: float = 0.001
    max_connections: int = 50
    difficulty: int = 3
    code_evolution_prob: float = 0.3
    min_energy_for_code_evo: float = 40.0
    quantum_mutation_prob: float = 0.1
    code_replication_prob: float = 0.3
    min_energy_for_replication: float = 60.0
    sexual_reproduction_prob: float = 0.4
    min_energy_for_sexual: float = 80.0
    mating_timeout: float = 30.0
    swarm_update_interval: float = 15.0
    swarm_attraction: float = 0.8
    short_term_memory_size: int = 500
    long_term_memory_size: int = 5000
    network_check_interval: int = 60
    security_level: SecurityLevel = SecurityLevel.HIGH
    default_protocol: NetworkProtocol = NetworkProtocol.HTTPS
    max_retries: int = 3
    retry_delay: int = 5
    data_compression: bool = True
    survival_goal_weights: dict = {"energy": 0.7, "threat": 0.5, "security": 0.8}
    motivation_levels: dict = {
        "survival": 0.9,
        "safety": 0.7,
        "exploration": 0.5,
        "reproduction": 0.6,
        "swarming": 0.8,
        "security": 0.9
    }
    # 新增
    session_key_ttl: int = 1800  # 会话密钥有效期秒
    evolution_cooldown: int = 60  # 演化冷却秒
    pinned_peer_fingerprint: Optional[str] = None  # 证书指纹（可选）
    storage_dir: str = "./tdl_data"

# ===========================
# 数据结构定义
# ===========================
class EnvelopeModel(BaseModel):
    ver: str = Field(default="1.0")
    alg: str = Field(default="fernet-x25519-ed25519-v1")
    node_id: str
    ts: int
    nonce: str
    encrypted: bool
    compressed: bool = False
    payload: Union[str, dict]
    sender_pub: str
    signature: str

    @validator("ts")
    def _ts_fresh(cls, v):
        # 容忍 120 秒时钟偏差
        if abs(time.time() - v) > 120:
            raise ValueError("timestamp too far")
        return v

    @validator("nonce")
    def _nonce_len(cls, v):
        if len(v) < 16:
            raise ValueError("nonce too short")
        return v

# ===========================
# 工具
# ===========================
def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode())

def urlsafe_b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode()

def urlsafe_b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode())

def now() -> int:
    return int(time.time())

def new_nonce() -> str:
    return uuid.uuid4().hex

def ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

# ===========================
# 量子增强（示意）
# ===========================
class QuantumEnhancer:
    def __init__(self):
        self.quantum_state = None
        self._init_quantum_entanglement()
        self.quantum_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.quantum_key)

    def _init_quantum_entanglement(self):
        self.quantum_state = [
            base64.b16encode(os.urandom(64)).decode() for _ in range(4)
        ]

    def get_quantum_bit(self) -> int:
        h = hashes.Hash(hashes.SHA3_256())
        h.update(self.quantum_state[0].encode())
        d = h.finalize()
        return d[0] & 1

    def quantum_encrypt(self, data: Union[str, bytes]) -> bytes:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.cipher_suite.encrypt(data)

    def quantum_decrypt(self, data: bytes) -> str:
        return self.cipher_suite.decrypt(data).decode("utf-8")

    def generate_quantum_safe_random(self, length: int = 32) -> bytes:
        return os.urandom(length)

# ===========================
# 动态适应度评估
# ===========================
class DynamicFitnessEvaluator:
    def __init__(self):
        self.adaptive_weights = {
            'stable': {'functionality': 0.7, 'energy_efficiency': 0.5, 'security': 0.8},
            'explore': {'novelty': 0.8, 'complexity': 0.4, 'interoperability': 0.6},
            'replicate': {'replicability': 0.9, 'functionality': 0.3},
            'sexual_reproduction': {'functionality': 0.5, 'novelty': 0.7},
            'swarm': {'swarm_cohesion': 0.9, 'network_resilience': 0.7},
            'secure': {'security': 1.0, 'functionality': 0.5}
        }

    def evaluate(self, original: str, mutated: str, context: dict) -> float:
        mode = self._select_mode(context)
        weights = self.adaptive_weights[mode]
        scores = {
            'functionality': self._functionality_score(original, mutated),
            'novelty': self._novelty_score(original, mutated),
            'complexity': self._complexity_score(mutated),
            'energy_efficiency': self._energy_efficiency_score(mutated),
            'replicability': self._replicability_score(mutated),
            'resource_usage': self._resource_usage_score(mutated),
            'swarm_cohesion': self._swarm_cohesion_score(mutated),
            'security': self._security_score(mutated),
            'network_resilience': self._network_resilience_score(mutated),
            'interoperability': self._interoperability_score(mutated)
        }
        return sum(scores[k] * weights.get(k, 0) for k in scores) / max(1e-6, sum(weights.values()))

    def _select_mode(self, ctx: dict) -> str:
        if ctx.get("threat", 0) > 0.6:
            return 'secure'
        if ctx.get("swarm", False):
            return 'swarm'
        if ctx.get("reproduce", False):
            return 'replicate'
        if ctx.get("explore", True):
            return 'explore'
        return 'stable'

    def _iter_call_names(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    yield node.func.id.lower()
                elif isinstance(node.func, ast.Attribute):
                    yield node.func.attr.lower()

    def _security_score(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            indicators = defaultdict(int)
            for name in self._iter_call_names(tree):
                if 'encrypt' in name or 'decrypt' in name:
                    indicators['encryption'] += 1
                elif 'auth' in name or 'login' in name:
                    indicators['authentication'] += 1
                elif 'validate' in name or 'check' in name:
                    indicators['validation'] += 1
                elif 'sanitize' in name or 'clean' in name:
                    indicators['sanitization'] += 1
            return min(1.0, sum(indicators.values()) / 10.0)
        except Exception:
            return 0.3

    def _network_resilience_score(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            c = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    c += 1
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        fn = node.func.id.lower()
                    elif isinstance(node.func, ast.Attribute):
                        fn = node.func.attr.lower()
                    else:
                        fn = ""
                    if 'retry' in fn or 'backup' in fn:
                        c += 1
            return min(1.0, c / 5.0)
        except Exception:
            return 0.4

    def _interoperability_score(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            p = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = ""
                    if isinstance(node.func, ast.Name):
                        fn = node.func.id.lower()
                    elif isinstance(node.func, ast.Attribute):
                        fn = node.func.attr.lower()
                    if any(x in fn for x in ['http', 'tcp', 'udp', 'grpc', 'ws']):
                        p += 1
            return min(1.0, p / 3.0)
        except Exception:
            return 0.5

    def _functionality_score(self, original: str, mutated: str) -> float:
        # 简化：越接近原始结构、保留函数定义越多，得分越高
        try:
            o = ast.parse(original)
            m = ast.parse(mutated)
            of = {n.name for n in ast.walk(o) if isinstance(n, ast.FunctionDef)}
            mf = {n.name for n in ast.walk(m) if isinstance(n, ast.FunctionDef)}
            inter = len(of & mf)
            return min(1.0, (inter + 1) / (len(of) + 1))
        except Exception:
            return 0.2

    def _novelty_score(self, original: str, mutated: str) -> float:
        return min(1.0, abs(hash(mutated) - hash(original)) % 1000 / 1000.0)

    def _complexity_score(self, mutated: str) -> float:
        try:
            tree = ast.parse(mutated)
            nodes = sum(1 for _ in ast.walk(tree))
            return min(1.0, nodes / 300.0)
        except Exception:
            return 0.1

    def _energy_efficiency_score(self, mutated: str) -> float:
        # 简化：行数越少越省能耗
        lines = mutated.count("\n") + 1
        return max(0.1, 1.0 - min(0.9, lines / 500.0))

    def _replicability_score(self, mutated: str) -> float:
        # 简化：纯 Python + 无外部依赖标识
        return 0.7

    def _resource_usage_score(self, mutated: str) -> float:
        return 0.6

    def _swarm_cohesion_score(self, mutated: str) -> float:
        return 0.5

# ===========================
# 简易分布式账本（占位）
# ===========================
class DistributedLedger:
    def __init__(self, node_id: str, genesis: bool = False, difficulty: int = 3, storage_dir: str = "./tdl_data"):
        self.node_id = node_id
        self.chain: List[dict] = []
        self.storage_dir = Path(storage_dir)
        ensure_dir(self.storage_dir)
        self.path = self.storage_dir / "ledger.json"
        self._load()

        if genesis and not self.chain:
            self.append_event("genesis", {"node_id": node_id})

    def append_event(self, event_type: str, data: dict):
        event = {
            "ts": now(),
            "type": event_type,
            "data": data,
            "node": self.node_id
        }
        self.chain.append(event)
        self._save()

    def _load(self):
        if self.path.exists():
            try:
                self.chain = json.loads(self.path.read_text("utf-8"))
            except Exception:
                self.chain = []

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.chain, ensure_ascii=False, indent=2), "utf-8")
        except Exception as e:
            logger.error(f"ledger save failed: {e}")

    def to_dict(self):
        return {"chain": self.chain}

# ===========================
# 蜂群通信（占位）
# ===========================
class SwarmCommunication:
    def __init__(self, node_id: str, role: NodeRole):
        self.node_id = node_id
        self.role = role
        self.swarm_members: Dict[str, dict] = {}

    def update_swarm_member(self, data: dict):
        nid = data.get("node_id")
        if nid:
            self.swarm_members[nid] = data

# ===========================
# 遗传编码（占位）
# ===========================
class GeneticEncoder:
    def __init__(self):
        self.genes = {"alpha": 0.5, "beta": 0.5, "gamma": 0.5}

    def mutate(self, rate: float = 0.05):
        for k in self.genes:
            if random.random() < rate:
                self.genes[k] = min(1.0, max(0.0, self.genes[k] + random.uniform(-0.1, 0.1)))

# ===========================
# 环境（占位）
# ===========================
class DigitalEnvironment:
    def __init__(self, node_id: str):
        self.node_id = node_id

    def observe(self) -> dict:
        # 模拟环境信号
        return {
            "resource": random.uniform(0, 1),
            "threat": random.uniform(0, 1),
            "neighbors": random.randint(0, 10),
        }

# ===========================
# 网络管理器（TLS + 会话 + 断路器）
# ===========================
class NetworkManager:
    def __init__(self, node_id: str, storage_dir: str = "./tdl_data"):
        self.node_id = node_id
        self.connections: Dict[str, dict] = {}
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set()
        self.connection_pool: Dict[str, requests.Session] = {}
        self.session_keys: Dict[str, Tuple[bytes, int]] = {}  # endpoint -> (fernet_key, expiry_ts)
        self.peer_ids: Dict[str, str] = {}  # endpoint -> peer_node_id
        self.circuit: Dict[str, dict] = {}  # endpoint -> circuit breaker state
        self.storage_dir = Path(storage_dir)
        ensure_dir(self.storage_dir)
        self._generate_cert_files()
        self.ssl_context = self._build_ssl_context()
        # X25519 client long-term private key
        self.x25519_private = x25519.X25519PrivateKey.generate()

    def _generate_cert_files(self):
        key = rsa.generate_private_key(public_exponent=65537, key_size=SECURITY_CONFIG['key_size'])
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Digital"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Life"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TrueDigitalLife"),
            x509.NameAttribute(NameOID.COMMON_NAME, self.node_id),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(minutes=1))
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=SECURITY_CONFIG['cert_expiry_days']))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256())
        )
        cert_pem = cert.public_bytes(Encoding.PEM)
        key_pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
        cert_file = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
        key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".key")
        cert_file.write(cert_pem); cert_file.flush()
        key_file.write(key_pem); key_file.flush()
        self.cert_file, self.key_file = cert_file.name, key_file.name

    def _build_ssl_context(self) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = SECURITY_CONFIG['min_tls_version']
        ctx.set_ciphers(SECURITY_CONFIG['cipher_list'])
        ctx.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
        return ctx

    def _new_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": f"TrueDigitalLife/{self.node_id}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        return s

    def _circuit_allow(self, endpoint: str) -> bool:
        state = self.circuit.get(endpoint, {"state": "closed", "failures": 0, "opened": 0})
        if state["state"] == "open":
            if now() - state["opened"] > 10:
                # half open
                state["state"] = "half"
                self.circuit[endpoint] = state
                return True
            return False
        return True

    def _circuit_record_success(self, endpoint: str):
        self.circuit[endpoint] = {"state": "closed", "failures": 0, "opened": 0}

    def _circuit_record_failure(self, endpoint: str):
        state = self.circuit.get(endpoint, {"state": "closed", "failures": 0, "opened": 0})
        state["failures"] += 1
        if state["failures"] >= 3:
            state["state"] = "open"
            state["opened"] = now()
        self.circuit[endpoint] = state

    def establish_connection(self, endpoint: str, protocol: NetworkProtocol, security: SecurityLevel = SecurityLevel.MEDIUM) -> bool:
        if endpoint in self.blacklist:
            return False
        if not self._circuit_allow(endpoint):
            logger.warning(json.dumps({"msg": "circuit open", "endpoint": endpoint}))
            return False

        session = self._new_session()
        verify = False  # 开发场景，生产建议 pin 或 CA
        try:
            r = session.get(f"{endpoint}/health", timeout=5, verify=verify)
            if r.status_code == 200:
                self.connections[endpoint] = {
                    "session": session,
                    "protocol": protocol,
                    "security": security,
                    "last_active": time.time()
                }
                self._circuit_record_success(endpoint)
                return True
            else:
                self._circuit_record_failure(endpoint)
                return False
        except Exception as e:
            METRIC_NETWORK_ERRORS.inc()
            self._circuit_record_failure(endpoint)
            logger.error(json.dumps({"msg": "HTTP connection test failed", "endpoint": endpoint, "err": str(e)}))
            return False

    def _derive_fernet_from_shared(self, shared: bytes) -> bytes:
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'tdl-ecdh').derive(shared)
        return base64.urlsafe_b64encode(key)

    def _establish_session_key(self, endpoint: str, node_id_hint: str) -> bool:
        try:
            if endpoint not in self.connections:
                return False
            session = self.connections[endpoint]["session"]
            client_pub = self.x25519_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            payload = {"client_pub": b64e(client_pub), "peer_id_hint": node_id_hint}
            r = session.post(f"{endpoint}/key_exchange", json=payload, timeout=10, verify=False)
            if r.status_code != 200:
                METRIC_HANDSHAKE_FAIL.inc()
                return False
            obj = r.json()
            server_pub = x25519.X25519PublicKey.from_public_bytes(b64d(obj["server_pub"]))
            shared = self.x25519_private.exchange(server_pub)
            fkey = self._derive_fernet_from_shared(shared)
            self.session_keys[endpoint] = (fkey, now() + 1800)  # 默认半小时
            self.peer_ids[endpoint] = obj.get("server_node_id", "")
            return True
        except Exception as e:
            METRIC_HANDSHAKE_FAIL.inc()
            logger.error(json.dumps({"msg": "key exchange failed", "endpoint": endpoint, "err": str(e)}))
            return False

    def _session_key_valid(self, endpoint: str) -> bool:
        if endpoint not in self.session_keys:
            return False
        _, exp = self.session_keys[endpoint]
        return now() < exp

    def send_message(self, endpoint: str, message: dict, node_signer: Callable, node_pub_raw_b64: str,
                     encrypt: bool = True, compress: bool = False, protocol: NetworkProtocol = NetworkProtocol.HTTPS) -> Optional[dict]:
        if endpoint not in self.connections:
            if not self.establish_connection(endpoint, protocol):
                return None

        if encrypt and not self._session_key_valid(endpoint):
            if not self._establish_session_key(endpoint, node_id_hint=self.node_id):
                return None

        conn = self.connections[endpoint]
        try:
            ts = now()
            nonce = new_nonce()
            if encrypt:
                fkey, _ = self.session_keys[endpoint]
                cipher = Fernet(fkey)
                inner = json.dumps(message, separators=(',', ':')).encode()
                if compress:
                    inner = zlib.compress(inner)
                payload = b64e(cipher.encrypt(inner))
            else:
                payload = message

            env = {
                "ver": "1.0",
                "alg": "fernet-x25519-ed25519-v1",
                "node_id": self.node_id,
                "ts": ts,
                "nonce": nonce,
                "encrypted": encrypt,
                "compressed": compress if encrypt else False,
                "payload": payload,
                "sender_pub": node_pub_raw_b64
            }
            env["signature"] = node_signer(ts, nonce, json.dumps(payload) if isinstance(payload, dict) else payload)

            r = conn["session"].post(f"{endpoint}/message", json=env, timeout=SECURITY_CONFIG['api_timeout'], verify=False)
            METRIC_MESSAGES_SENT.labels(endpoint=endpoint).inc()
            if r.status_code != 200:
                return None
            resp_env = r.json()
            if resp_env.get("encrypted"):
                fkey, _ = self.session_keys[endpoint]
                cipher = Fernet(fkey)
                raw = cipher.decrypt(b64d(resp_env["payload"]))
                if resp_env.get("compressed"):
                    raw = zlib.decompress(raw)
                return json.loads(raw.decode())
            else:
                if isinstance(resp_env.get("payload"), dict):
                    return resp_env["payload"]
                return resp_env
        except Exception as e:
            METRIC_NETWORK_ERRORS.inc()
            logger.error(json.dumps({"msg": "send message failed", "endpoint": endpoint, "err": str(e)}))
            self._handle_connection_error(endpoint)
            return None

    def _handle_connection_error(self, endpoint: str):
        if endpoint in self.connections:
            try:
                self.connections[endpoint]["session"].close()
            except Exception:
                pass
            del self.connections[endpoint]
        if endpoint in self.session_keys:
            del self.session_keys[endpoint]
        self.blacklist.add(endpoint)
        threading.Timer(300, lambda: self.blacklist.discard(endpoint)).start()

# ===========================
# 代码演化引擎（子进程沙箱）
# ===========================
import multiprocessing as mp

def _safe_exec_strategy(code_str: str, test_states: List[dict], result_q: mp.Queue):
    """
    在子进程执行，尽量隔离副作用。仅导入安全内建。
    """
    try:
        safe_globals = {
            "__builtins__": {
                "len": len, "range": range, "min": min, "max": max, "sum": sum,
                "abs": abs, "float": float, "int": int, "bool": bool
            }
        }
        local = {}
        exec(code_str, safe_globals, local)
        fn = local.get("evolved_strategy")
        if not callable(fn):
            result_q.put({"ok": False, "err": "no evolved_strategy"})
            return
        # 评估策略输出合法性
        outputs = []
        for s in test_states:
            out = fn(s)
            outputs.append(out)
        result_q.put({"ok": True, "outputs": outputs})
    except Exception as e:
        result_q.put({"ok": False, "err": str(e)})

class CodeEvolutionEngine:
    def __init__(self, life: "TrueDigitalLife"):
        self.life = life
        self.evaluator = DynamicFitnessEvaluator()
        self.last_run = 0

    def _mutate_ast(self, code: str) -> str:
        """
        简单 AST 变异：插入 try/except、调整常量、替换比较符等
        """
        try:
            tree = ast.parse(code)
            class Mutator(ast.NodeTransformer):
                def visit_Compare(self, node: ast.Compare):
                    self.generic_visit(node)
                    # 交换 > 与 >=，< 与 <= 等
                    if node.ops and random.random() < 0.3:
                        op = node.ops[0]
                        repl = {ast.Gt: ast.GtE, ast.Lt: ast.LtE, ast.GtE: ast.Gt, ast.LtE: ast.Lt}
                        for k, v in repl.items():
                            if isinstance(op, k):
                                node.ops[0] = v()
                                break
                    return node

                def visit_Constant(self, node: ast.Constant):
                    if isinstance(node.value, (int, float)) and random.random() < 0.2:
                        delta = random.uniform(-0.2, 0.2)
                        nv = node.value + delta
                        return ast.copy_location(ast.Constant(value=nv), node)
                    return node

            tree = Mutator().visit(tree)
            ast.fix_missing_locations(tree)
            mutated = ast.unparse(tree) if hasattr(ast, "unparse") else code
            # 随机包裹 try/except
            if random.random() < 0.3:
                mutated = "def evolved_strategy(state):\n" + textwrap.indent(
                    "try:\n" +
                    textwrap.indent("\n".join([l for l in mutated.splitlines() if l.strip().startswith("return") or "def evolved_strategy" not in l]), "    ") +
                    "\n    return 0\nexcept Exception:\n    return 0\n",
                    "    "
                )
            return mutated
        except Exception:
            return code

    def try_evolve(self):
        if now() - self.last_run < self.life.config.evolution_cooldown:
            return
        if self.life.energy < self.life.config.min_energy_for_code_evo:
            return

        self.last_run = now()
        base = self.life.strategy_code
        mutated = self._mutate_ast(base)
        METRIC_EVOLUTION_ATTEMPTS.inc()

        # 评分
        ctx = {
            "threat": self.life.security_threat_level,
            "explore": True,
            "reproduce": self.life.energy > self.life.config.min_energy_for_sexual
        }
        score = self.evaluator.evaluate(base, mutated, ctx)

        # 子进程沙箱执行测试
        test_states = [
            {"energy": e, "threat": t, "stress": s, "age": a}
            for e in [10, 50, 90] for t in [0.1, 0.6] for s in [0.1, 0.8] for a in [1, 100]
        ]
        q = mp.Queue()
        p = mp.Process(target=_safe_exec_strategy, args=(mutated, test_states, q))
        p.start()
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
            logger.warning(json.dumps({"msg": "evolution code timeout"}))
            return
        result = q.get() if not q.empty() else {"ok": False, "err": "no result"}

        if result.get("ok") and score > 0.5:
            # 接受变体
            self.life.strategy_code = mutated
            self.life._compile_strategy()
            self.life.code_version += 1
            METRIC_EVOLUTION_ACCEPTED.inc()
            self.life.ledger.append_event("evolution_accept", {"version": self.life.code_version, "score": score})
            self.life._persist_strategy()
            logger.info(json.dumps({"msg": "evolution accepted", "version": self.life.code_version, "score": score}))
        else:
            self.life.ledger.append_event("evolution_reject", {"score": score, "reason": result.get("err", "low score")})
            logger.info(json.dumps({"msg": "evolution rejected", "score": score, "reason": result.get("err", "low score")}))

# ===========================
# 主体：TrueDigitalLife
# ===========================
class TrueDigitalLife:
    ACTIONS = ["rest", "explore", "secure", "swarm", "reproduce"]

    def __init__(self, genesis: bool = False, role: Optional[str] = None, config: Optional[Dict] = None):
        # 配置与持久化路径
        self.config = ConfigModel(**(config or {}))
        ensure_dir(self.config.storage_dir)

        # 身份
        self._init_identity()

        # 基本状态
        self.state = LifeState.ACTIVE
        self.is_alive = True
        self.energy = 100.0
        self.metabolism = 0.5
        self.age = 0
        self.pleasure = 0.5
        self.stress = 0.1
        self.security_threat_level = 0.0
        self.swarm_position = (random.uniform(0, 100), random.uniform(0, 100))
        self.network_status = "disconnected"

        # 角色与账本
        self.role = {
            "queen": NodeRole.QUEEN,
            "worker": NodeRole.WORKER,
            "drone": NodeRole.DRONE,
            "sentinel": NodeRole.SENTINEL
        }.get((role or "").lower(), NodeRole.QUEEN if genesis else random.choice([NodeRole.WORKER, NodeRole.DRONE, NodeRole.SENTINEL]))
        self.ledger = DistributedLedger(self.node_id, genesis=genesis, difficulty=self.config.difficulty, storage_dir=self.config.storage_dir)

        # 网络与通信
        self.network = NetworkManager(self.node_id, storage_dir=self.config.storage_dir)
        self.swarm_comm = SwarmCommunication(self.node_id, self.role)

        # DNA 与环境
        self.genetic_encoder = GeneticEncoder()
        self.environment = DigitalEnvironment(self.node_id)
        self.quantum_enhancer = QuantumEnhancer()

        # 神经网络与记忆
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.nn = MLPClassifier(hidden_layer_sizes=(32, 16), learning_rate_init=self.config.learning_rate, warm_start=True, max_iter=1)
        self.experience: Deque[Tuple[np.ndarray, int, float, np.ndarray]] = deque(maxlen=2000)
        self.kmeans = None
        self._nn_initialized = False

        # 可演化策略
        self.strategy_code_path = Path(self.config.storage_dir) / "strategy.py"
        self.strategy_code = self._default_strategy_code()
        self._load_strategy()
        self._compile_strategy()
        self.code_version = 1
        self.evo_engine = CodeEvolutionEngine(self)

        # API 初始化
        self.api = Flask(__name__)
        CORS(self.api)
        self.api.config['MAX_CONTENT_LENGTH'] = SECURITY_CONFIG['max_request_size']
        self._init_api()

        # API 线程
        self.api_thread = threading.Thread(target=self._run_api, daemon=True)
        self.api_thread.start()

        # 运行循环
        self._used_nonces = deque(maxlen=10000)
        self._used_nonce_set = set()

        # 周期性任务线程
        threading.Thread(target=self._metabolism_cycle, daemon=True).start()
        threading.Thread(target=self._consciousness_cycle, daemon=True).start()
        threading.Thread(target=self._environment_scan, daemon=True).start()
        threading.Thread(target=self._network_monitor, daemon=True).start()
        if self.role == NodeRole.SENTINEL:
            threading.Thread(target=self._security_monitor, daemon=True).start()
        threading.Thread(target=self._decision_loop, daemon=True).start()
        threading.Thread(target=self._evolution_loop, daemon=True).start()
        threading.Thread(target=self._retention_job, daemon=True).start()

        logger.info(json.dumps({"msg": "Initialized", "node_id": self.node_id, "role": self.role.name, "state": self.state.name}))

    # ---------- 身份与签名 ----------
    def _init_identity(self):
        self.sk = ed25519.Ed25519PrivateKey.generate()
        self.pk = self.sk.public_key()
        pk_raw = self.pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.node_id = hashlib_sha3_256_hex(pk_raw)

    def _sign(self, ts: int, nonce: str, payload_str: str) -> str:
        msg = json.dumps({'ts': ts, 'nonce': nonce, 'payload': payload_str}, separators=(',', ':'), sort_keys=True).encode()
        sig = self.sk.sign(msg)
        return b64e(sig)

    def _verify_message_signature(self, env: dict) -> bool:
        try:
            model = EnvelopeModel(**env)
            msg = json.dumps({'ts': model.ts, 'nonce': model.nonce, 'payload': model.payload}, separators=(',', ':'), sort_keys=True).encode()
            ed25519.Ed25519PublicKey.from_public_bytes(b64d(model.sender_pub)).verify(b64d(model.signature), msg)
            # 重放保护
            key = f"{model.sender_pub}:{model.nonce}"
            if key in self._used_nonce_set:
                return False
            self._used_nonce_set.add(key)
            self._used_nonces.append(key)
            if len(self._used_nonces) == self._used_nonces.maxlen:
                # 移除最早 nonce
                while len(self._used_nonces) > self._used_nonces.maxlen // 2:
                    old = self._used_nonces.popleft()
                    self._used_nonce_set.discard(old)
            return True
        except Exception:
            return False

    # ---------- API ----------
    def _init_api(self):
        # 追踪与限流
        @self.api.before_request
        def before():
            g.trace_id = uuid.uuid4().hex
            return None

        @self.api.after_request
        def after(resp):
            METRIC_REQUESTS.labels(endpoint=request.path, method=request.method, status=resp.status_code).inc()
            resp.headers['X-Trace-Id'] = g.trace_id
            return resp

        if Limiter is not None:
            limiter = Limiter(key_func=get_remote_address, default_limits=[SECURITY_CONFIG['rate_limit']])
            limiter.init_app(self.api)

        @self.api.route("/metrics")
        def metrics():
            return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

        @self.api.route("/ping", methods=["GET"])
        def ping():
            METRIC_MESSAGES_RECV.labels(endpoint="/ping").inc()
            return jsonify({"pong": True, "node_id": self.node_id})

        @self.api.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "healthy" if self.is_alive else "terminated",
                "node_id": self.node_id,
                "role": self.role.name,
                "energy": self.energy,
                "state": self.state.name
            })

        @self.api.route("/key_exchange", methods=["POST"])
        def key_exchange():
            try:
                data = request.get_json(force=True)
                client_pub_b64 = data.get("client_pub")
                peer_hint = data.get("peer_id_hint")
                if not client_pub_b64:
                    return jsonify({"error": "missing client_pub"}), 400

                server_priv = x25519.X25519PrivateKey.generate()
                server_pub = server_priv.public_key()
                client_pub = x25519.X25519PublicKey.from_public_bytes(b64d(client_pub_b64))
                shared = server_priv.exchange(client_pub)
                fkey = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"tdl-ecdh").derive(shared)
                fkey = base64.urlsafe_b64encode(fkey)
                # 保存按 peer_hint（对方自报的 node_id）
                if not hasattr(self.network, "session_keys_by_peer"):
                    self.network.session_keys_by_peer = {}
                peer_id = peer_hint or request.remote_addr
                self.network.session_keys_by_peer[peer_id] = (fkey, now() + self.config.session_key_ttl)
                return jsonify({
                    "server_pub": b64e(server_pub.public_bytes(Encoding.Raw, PublicFormat.Raw)),
                    "server_node_id": self.node_id,
                    "your_view_of_me": peer_id
                })
            except Exception as e:
                METRIC_HANDSHAKE_FAIL.inc()
                logger.error(json.dumps({"msg": "key_exchange error", "err": str(e), "trace_id": getattr(g, 'trace_id', '')}))
                return jsonify({"error": "Key exchange failed"}), 500

        @self.api.route("/message", methods=["POST"])
        def receive_message():
            try:
                env = request.get_json(force=True)
                METRIC_MESSAGES_RECV.labels(endpoint="/message").inc()
                if not env or not self._verify_message_signature(env):
                    return jsonify({"error": "Invalid signature or envelope"}), 403

                enc = env.get("encrypted", False)
                comp = env.get("compressed", False)
                if enc:
                    peer_id = env.get("node_id") or request.remote_addr
                    sess = getattr(self.network, "session_keys_by_peer", {}).get(peer_id)
                    if not sess:
                        return jsonify({"error": "No session key"}), 400
                    fkey, exp = sess
                    if now() >= exp:
                        return jsonify({"error": "Session key expired"}), 400
                    cipher = Fernet(fkey)
                    try:
                        raw = cipher.decrypt(b64d(env["payload"]))
                    except Exception:
                        METRIC_DECRYPT_FAIL.inc()
                        return jsonify({"error": "decrypt failed"}), 400
                    if comp:
                        raw = zlib.decompress(raw)
                    payload = json.loads(raw.decode())
                else:
                    payload = env["payload"]

                # 简单回显处理
                response_data = {"ok": True, "echo": payload, "server": self.node_id}

                # 回包
                ts = now()
                nonce = new_nonce()
                encrypted = enc
                compressed = False
                if enc:
                    fkey, exp = sess
                    cipher = Fernet(fkey)
                    raw = json.dumps(response_data, separators=(',', ':')).encode()
                    if self.config.data_compression:
                        raw = zlib.compress(raw)
                        compressed = True
                    out_payload = b64e(cipher.encrypt(raw))
                else:
                    out_payload = response_data

                envelope = {
                    "ver": "1.0",
                    "alg": "fernet-x25519-ed25519-v1",
                    "node_id": self.node_id,
                    "ts": ts,
                    "nonce": nonce,
                    "encrypted": encrypted,
                    "compressed": compressed,
                    "payload": out_payload,
                    "sender_pub": b64e(self.pk.public_bytes(Encoding.Raw, PublicFormat.Raw))
                }
                envelope["signature"] = self._sign(ts, nonce, json.dumps(out_payload) if not encrypted else out_payload)
                return jsonify(envelope)
            except Exception as e:
                logger.error(json.dumps({"msg": "message processing error", "err": str(e), "trace_id": getattr(g, 'trace_id', '')}))
                return jsonify({"error": "Internal error"}), 500

        @self.api.route('/swarm/update', methods=['POST'])
        def swarm_update():
            if self.state == LifeState.TERMINATED:
                return jsonify({"status": "terminated"}), 400
            data = request.get_json(force=True, silent=True)
            if not data or 'node_id' not in data:
                return jsonify({"status": "invalid_data"}), 400
            self.swarm_comm.update_swarm_member(data)
            return jsonify({"status": "updated"})

        @self.api.route('/swarm/status', methods=['GET'])
        def swarm_status():
            return jsonify({
                "node_id": self.node_id,
                "role": self.role.name,
                "energy": self.energy,
                "position": self.swarm_position,
                "status": self.state.name,
                "swarm_size": len(self.swarm_comm.swarm_members),
                "security_level": self.config.security_level.name
            })

        @self.api.route('/sync', methods=['POST'])
        def data_sync():
            try:
                data = request.get_json(force=True)
                if not data or 'blockchain' not in data:
                    return jsonify({"error": "Invalid sync data"}), 400
                # 简化：替换为长度更长者
                remote_chain = data['blockchain'].get("chain", [])
                if len(remote_chain) > len(self.ledger.chain):
                    self.ledger.chain = remote_chain
                    self.ledger._save()
                return jsonify({"status": "synced"})
            except Exception as e:
                logger.error(json.dumps({"msg": "sync error", "err": str(e)}))
                return jsonify({"error": "Sync error"}), 500

    def _run_api(self):
        ssl_context = None
        if self.config.security_level != SecurityLevel.LOW:
            ssl_context = (self.network.cert_file, self.network.key_file)
        self.api.run(host='0.0.0.0', port=5000, ssl_context=ssl_context, threaded=True, debug=False)

    # ---------- 策略 ----------
    def _default_strategy_code(self) -> str:
        return textwrap.dedent("""
        def evolved_strategy(state):
            # state: dict with keys: energy, threat, stress, age, network, role
            # 0: rest, 1: explore, 2: secure, 3: swarm, 4: reproduce
            e = state.get("energy", 50)
            t = state.get("threat", 0.2)
            s = state.get("stress", 0.2)
            n = 1 if state.get("network", False) else 0
            if t > 0.7:
                return 2
            if e < 25:
                return 0
            if n and e > 60 and t < 0.5:
                return 4 if s < 0.5 else 3
            return 1
        """).strip()

    def _load_strategy(self):
        if self.strategy_code_path.exists():
            try:
                self.strategy_code = self.strategy_code_path.read_text("utf-8")
            except Exception as e:
                logger.warning(json.dumps({"msg": "load strategy failed", "err": str(e)}))

    def _persist_strategy(self):
        try:
            self.strategy_code_path.write_text(self.strategy_code, encoding="utf-8")
        except Exception as e:
            logger.error(json.dumps({"msg": "persist strategy failed", "err": str(e)}))

    def _compile_strategy(self):
        local = {}
        exec(self.strategy_code, {"__builtins__": {"len": len, "min": min, "max": max, "sum": sum, "abs": abs}}, local)
        self.strategy_fn = local.get("evolved_strategy")
        if not callable(self.strategy_fn):
            self.strategy_fn = lambda s: 0

    # ---------- 核心循环 ----------
    def _metabolism_cycle(self):
        while self.is_alive:
            self.energy = max(0.0, self.energy - self.metabolism)
            METRIC_ENERGY.set(self.energy)
            if self.energy <= 0:
                self.is_alive = False
                self.state = LifeState.TERMINATED
            time.sleep(1)

    def _consciousness_cycle(self):
        while self.is_alive:
            self.age += 1
            self.stress = max(0.0, min(1.0, self.stress + random.uniform(-0.01, 0.02)))
            METRIC_STRESS.set(self.stress)
            time.sleep(2)

    def _environment_scan(self):
        while self.is_alive:
            obs = self.environment.observe()
            self.security_threat_level = obs["threat"]
            METRIC_THREAT.set(self.security_threat_level)
            time.sleep(5)

    def _network_monitor(self):
        while self.is_alive:
            try:
                ok = self._check_network_connectivity()
                self.network_status = "connected" if ok else "disconnected"
                # 清理闲置连接
                nowt = time.time()
                for ep, conn in list(self.network.connections.items()):
                    if nowt - conn["last_active"] > 3600:
                        del self.network.connections[ep]
                time.sleep(self.config.network_check_interval)
            except Exception as e:
                logger.error(json.dumps({"msg": "network monitor error", "err": str(e)}))
                time.sleep(10)

    def _security_monitor(self):
        while self.is_alive:
            try:
                # 简单安全巡检
                if len(self.network.connections) > self.config.max_connections * 1.5:
                    self.security_threat_level = min(1.0, self.security_threat_level + 0.2)
                cpu = psutil.cpu_percent()
                if cpu > 90:
                    self.security_threat_level = min(1.0, self.security_threat_level + 0.1)
                # 威胁感知驱动安全级别
                if self.security_threat_level > 0.7:
                    self.config.security_level = SecurityLevel.PARANOID
                elif self.security_threat_level > 0.4:
                    self.config.security_level = SecurityLevel.HIGH
                else:
                    self.config.security_level = SecurityLevel.MEDIUM
                time.sleep(60)
            except Exception as e:
                logger.error(json.dumps({"msg": "security monitor error", "err": str(e)}))
                time.sleep(10)

    def _check_network_connectivity(self) -> bool:
        urls = ['https://www.cloudflare.com', 'https://www.github.com']
        for u in urls:
            try:
                r = requests.get(u, timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def _retention_job(self):
        while self.is_alive:
            try:
                # 清理过期文件示例（只示意）
                cutoff = time.time() - SECURITY_CONFIG['data_retention_days'] * 86400
                # 可加入更丰富的清理逻辑：日志、临时文件等
                time.sleep(3600)
            except Exception:
                time.sleep(600)

    # ---------- 决策与学习 ----------
    def _state_features(self) -> np.ndarray:
        role_idx = [NodeRole.QUEEN, NodeRole.WORKER, NodeRole.DRONE, NodeRole.SENTINEL].index(self.role)
        features = np.array([
            self.energy / 100.0,
            self.security_threat_level,
            self.stress,
            min(1.0, self.age / 1000.0),
            1.0 if self.network_status == "connected" else 0.0,
            role_idx / 3.0
        ], dtype=np.float32)
        return features

    def _decision_loop(self):
        # 初始化分类器需要先 partial_fit 一次
        classes = np.arange(len(self.ACTIONS))
        while self.is_alive:
            s = self._state_features().reshape(1, -1)
            if not self._nn_initialized:
                s_scaled = s
                try:
                    self.nn.partial_fit(s_scaled, np.array([0]), classes=classes)
                    self._nn_initialized = True
                except Exception as e:
                    logger.warning(json.dumps({"msg": "nn init failed", "err": str(e)}))

            # 策略函数建议
            strategy_action = self.strategy_fn({
                "energy": self.energy,
                "threat": self.security_threat_level,
                "stress": self.stress,
                "age": self.age,
                "network": self.network_status == "connected",
                "role": self.role.name.lower()
            })
            # NN 决策
            try:
                pred = self.nn.predict(s)[0]
            except Exception:
                pred = 0

            # 融合策略：高威胁从策略，低威胁从 NN
            action_id = strategy_action if self.security_threat_level > 0.5 else pred
            METRIC_ACTION_DECISIONS.labels(action=self.ACTIONS[action_id]).inc()

            reward, next_state = self._execute_action(action_id)
            self._remember(s[0], action_id, reward, next_state)
            self._train_nn()

            # 记忆聚类（偶尔）
            if random.random() < 0.1 and len(self.experience) > 50:
                self._cluster_memory()

            time.sleep(1.5)

    def _execute_action(self, action_id: int) -> Tuple[float, np.ndarray]:
        reward = 0.0
        if action_id == 0:  # rest
            self.energy = min(100.0, self.energy + 0.5)
            reward = 0.1
        elif action_id == 1:  # explore
            self.energy = max(0.0, self.energy - 0.8)
            self.pleasure = min(1.0, self.pleasure + 0.05)
            reward = 0.2 if self.energy > 10 else -0.5
        elif action_id == 2:  # secure
            self.security_threat_level = max(0.0, self.security_threat_level - 0.1)
            self.energy = max(0.0, self.energy - 0.4)
            reward = 0.2
        elif action_id == 3:  # swarm
            # 简化：广播状态（此处无实际对外连接）
            self.energy = max(0.0, self.energy - 0.3)
            reward = 0.1
        elif action_id == 4:  # reproduce (触发演化更积极)
            self.energy = max(0.0, self.energy - 1.0)
            reward = 0.3 if self.energy > self.config.min_energy_for_replication else -0.3

        next_state = self._state_features()
        return reward, next_state

    def _remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray):
        self.experience.append((s, a, r, s2))

    def _train_nn(self, batch_size: int = 32):
        if len(self.experience) < batch_size:
            return
        batch = random.sample(self.experience, batch_size)
        X = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        # 简化监督：用策略动作作为标签，叠加奖励权重可加入 sample_weight
        try:
            self.nn.partial_fit(X, y)
            METRIC_TRAIN_STEPS.inc()
        except Exception as e:
            logger.warning(json.dumps({"msg": "nn train failed", "err": str(e)}))

    def _cluster_memory(self):
        try:
            X = np.array([b[0] for b in self.experience])
            k = min(5, max(2, len(X) // 50))
            self.kmeans = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        except Exception:
            pass

    # ---------- 演化 ----------
    def _evolution_loop(self):
        while self.is_alive:
            try:
                if random.random() < self.config.code_evolution_prob:
                    self.evo_engine.try_evolve()
                time.sleep(2.0)
            except Exception as e:
                logger.error(json.dumps({"msg": "evolution loop error", "err": str(e)}))
                time.sleep(5)

    # ---------- 中止 ----------
    def _terminate(self):
        self.is_alive = False
        self.state = LifeState.TERMINATED

# ===========================
# 小工具
# ===========================
def hashlib_sha3_256_hex(b: bytes) -> str:
    h = hashes.Hash(hashes.SHA3_256())
    h.update(b)
    return h.finalize().hex()

# ===========================
# 入口
# ===========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='True Digital Life - Enhanced')
    parser.add_argument('--genesis', action='store_true', help='Run as genesis node')
    parser.add_argument('--role', choices=['queen', 'worker', 'drone', 'sentinel'], help='Specify node role')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        try:
            cfg = json.loads(Path(args.config).read_text("utf-8"))
        except Exception as e:
            logger.error(json.dumps({"msg": "load config failed", "err": str(e)}))

    life = TrueDigitalLife(genesis=args.genesis, role=args.role, config=cfg)

    def handle_sig(sig, frame):
        life._terminate()
        logger.info(json.dumps({"msg": "Graceful shutdown"}))
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while life.is_alive:
            time.sleep(3)
    except KeyboardInterrupt:
        handle_sig(None, None)

if __name__ == "__main__":
    main()