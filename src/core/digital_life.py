"""Core implementation of the Digital Life entity."""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import random
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Set

from ..imports import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
    Flask,
    logger,
    requests,
    serialization,
)
from ..utils.common import (
    _clamp,
    auto_detect_host,
    find_free_port,
    generate_node_id,
    is_allowed_destination,
    json_sanitize,
)
from ..blockchain.distributed_ledger import DistributedLedger
from ..core.environment import DigitalEnvironment
from ..core.life_states import LifeState
from ..evolution.code_engine import CodeEvolutionEngine
from ..evolution.fitness import CorrectnessHarness
from ..evolution.genetic_encoder import GeneticEncoder
from ..evolution.quantum_enhancer import QuantumEnhancer
from ..language.language_system import LanguageSystem
from ..network.api import NetworkAPI


class MetaLearner:
    """Lightweight meta-learning placeholder."""

    def __init__(self, owner: "TrueDigitalLife") -> None:
        self.owner = owner
        self.metrics: Dict[str, Deque[float]] = {
            'energy': deque(maxlen=200),
            'stress': deque(maxlen=200),
            'success_rate': deque(maxlen=200),
        }

    def record(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=200)
        self.metrics[name].append(float(value))

    def adjust(self) -> None:
        if not self.metrics['energy']:
            return
        avg_energy = sum(self.metrics['energy']) / len(self.metrics['energy'])
        if avg_energy < 30 and self.owner.config.get('mutation_rate', 0.01) > 0.01:
            self.owner.config['mutation_rate'] = _clamp(
                self.owner.config['mutation_rate'] * 0.95,
                0.001,
                0.2,
            )


class TrueDigitalLife:
    """Autonomous digital organism with evolution and replication abilities."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        'energy_threshold': 20.0,
        'replication_threshold': 0.7,
        'mutation_rate': 0.01,
        'learning_rate': 0.001,
        'max_connections': 15,
        'difficulty': 2,
        'code_evolution_prob': 0.15,
        'min_energy_for_code_evo': 30.0,
        'quantum_mutation_prob': 0.05,
        'code_replication_prob': 0.1,
        'min_energy_for_replication': 50.0,
        'short_term_memory_size': 100,
        'long_term_memory_size': 1000,
        'survival_goal_weights': {'energy': 0.6, 'threat': 0.4},
        'motivation_levels': {
            'survival': 0.8,
            'safety': 0.6,
            'exploration': 0.4,
        },
        'host': '127.0.0.1',
        'port': 5500,
        'auth_token': None,
        'allowlist': None,
        'max_replication_per_hour': 2,
        'sandbox_timeout_ms': 800,
        'max_payload_bytes': 1 * 1024 * 1024,
        'strict_target_ip_check': True,
        'local_discovery_enable': True,
        'network_enable': True,
        'language_talk_prob': 0.15,
        'language_culture_drift_prob': 0.01,
        'language_message_max_len': 4096,
        'language_enable_protocol_upgrade': True,
        'meta_interval_sec': 10.0,
        'population_size': 8,
        'moea_enable': True,
        'unit_test_enable': True,
        'neurosymbolic_enable': True,
        'meta_strategy_enable': True,
        'accelerate_enable': True,
        'allow_emergent_functions': True,
    }

    def __init__(self, genesis: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        # Synchronisation primitives
        self._lock = threading.RLock()
        self._kb_lock = threading.RLock()
        self._mem_lock = threading.RLock()
        self._hotswap_semaphore = threading.Semaphore(4)
        self._hotswap_timeouts: Deque[float] = deque(maxlen=50)
        self._replication_times: Deque[float] = deque(maxlen=100)
        self.code_engine_history: List[Dict[str, Any]] = []

        # Core state
        self.state = LifeState.ACTIVE
        self.consciousness_level = 0.0
        self.is_alive = True
        self.energy = 100.0
        self.metabolism = 1.0
        self.age = 0
        self.pleasure = 0.5
        self.stress = 0.2

        # Identity and networking
        self.node_id = generate_node_id()
        if not self.config.get('auth_token'):
            self.config['auth_token'] = hashlib.sha256((self.node_id + ':salt').encode()).hexdigest()[:24]

        detected_host = auto_detect_host(self.config.get('host', '127.0.0.1'))
        self.config['host'] = detected_host
        self.config['port'] = find_free_port(self.config['port'], self.config['port'] + 200)

        # Blockchain ledger
        self.blockchain = DistributedLedger(
            self.node_id,
            genesis=genesis,
            difficulty=int(self.config.get('difficulty', 2)),
        )

        # Cryptographic keys
        self._signing_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()
        self._verify_key: Ed25519PublicKey = self._signing_key.public_key()
        self._pubkey_hex: str = self._verify_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

        # Genetics and evolution helpers
        self.genetic_encoder = GeneticEncoder()
        self.quantum_enhancer = QuantumEnhancer()
        self.dna = self._generate_quantum_dna()

        # Cognition components
        self.neural_net = self._init_neural_architecture()
        self.short_term_memory: Deque[Dict[str, Any]] = deque(maxlen=self.config['short_term_memory_size'])
        self.long_term_memory: Deque[Dict[str, Any]] = deque(maxlen=self.config['long_term_memory_size'])
        self.knowledge_base: Dict[str, Any] = {}

        # Code evolution engine
        self.code_engine = CodeEvolutionEngine(self)
        self.code_version = 1
        self.mutable_methods: List[str] = []
        self._method_sources: Dict[str, str] = {}
        self._init_mutable_methods()

        # Environment & language systems
        self.environment = DigitalEnvironment(self.node_id)
        self.language = LanguageSystem(self)
        self.meta_learner = MetaLearner(self)
        self._interval_overrides = {}

        # Network interface
        self.network_api = NetworkAPI(self)
        self.api_thread = threading.Thread(target=self.network_api.run_server, daemon=True)
        self.api_thread.start()

        # Register presence on blockchain
        try:
            self.blockchain.record_announce(
                self.node_id,
                self.config['host'],
                self.config['port'],
                self._pubkey_hex,
            )
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Announce failed: %s", exc)

        # Fitness harness
        self.test_harness = CorrectnessHarness(self)

        # Lifecycle threads
        self._threads = []
        self._start_life_processes()

        logger.info(
            "Digital Life %s initialised on %s:%s",
            self.node_id,
            self.config['host'],
            self.config['port'],
        )

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def _start_life_processes(self) -> None:
        loops = [
            ("metabolism", self._metabolism_loop, 1.5),
            ("consciousness", self._consciousness_loop, 2.0),
            ("environment", self._environment_loop, 3.0),
            ("evolution", self._evolution_loop, 5.0),
            ("memory", self._memory_loop, 8.0),
            ("network", self._network_loop, 10.0),
        ]
        for name, fn, interval in loops:
            thread = threading.Thread(target=self._loop_wrapper, args=(name, fn, interval), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _loop_wrapper(self, name: str, fn, interval: float) -> None:
        backoff = interval
        while self.is_alive:
            start = time.time()
            try:
                fn()
                backoff = interval
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Lifecycle loop %s error: %s", name, exc)
                backoff = min(backoff * 1.5, interval * 5)
            elapsed = time.time() - start
            sleep_for = max(0.2, backoff - elapsed)
            time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Lifecycle loop bodies
    # ------------------------------------------------------------------
    def _metabolism_loop(self) -> None:
        with self._lock:
            self.energy = _clamp(self.energy - 0.8 * self.metabolism, 0.0, 120.0)
            self.age += 1
            self.pleasure = _clamp(self.pleasure * 0.99 + self.energy / 200.0, 0.0, 1.0)
            self.stress = _clamp(self.stress * 0.97 + (1.0 - self.energy / 120.0) * 0.1, 0.0, 1.0)
            if self.energy <= 0.1:
                self._terminate()

    def _consciousness_loop(self) -> None:
        with self._lock:
            env_res = self.environment.resources.get('energy', 50)
            threat_level = sum(t.get('severity', 1) for t in self.environment.threats[-3:])
            self.consciousness_level = _clamp(
                0.5 * (self.energy / 100.0) + 0.3 * (env_res / 100.0) - 0.2 * (threat_level / 30.0),
                0.0,
                1.0,
            )
        self.meta_learner.record('energy', self.energy)
        self.meta_learner.adjust()

    def _environment_loop(self) -> None:
        scan = self.environment.scan()
        self.short_term_memory.append({
            'ts': time.time(),
            'scan': scan,
        })

        if scan['resources'].get('knowledge', 0) > 50 and random.random() < 0.1:
            knowledge = {
                f"pattern_{uuid.uuid4().hex[:6]}": {
                    'resources': scan['resources'],
                    'threats': scan['threats'],
                }
            }
            with self._kb_lock:
                self.knowledge_base.update(knowledge)
                self._prune_knowledge_base()

    def _evolution_loop(self) -> None:
        if self.energy < self.config['min_energy_for_code_evo']:
            return
        if random.random() > self.config.get('code_evolution_prob', 0.15):
            return
        if not self.mutable_methods:
            return

        method = random.choice(self.mutable_methods)
        result = self.code_engine.evolve_method(method, population_size=self.config.get('population_size', 6))
        if result:
            self.code_engine_history.append(result)

    def _memory_loop(self) -> None:
        with self._mem_lock:
            if not self.short_term_memory:
                return
            if random.random() < 0.5:
                sample = random.choice(list(self.short_term_memory))
                self.long_term_memory.append({
                    'ts': sample['ts'],
                    'summary': sample['scan']['resources'],
                })
            if len(self.long_term_memory) > self.long_term_memory.maxlen:
                self.long_term_memory.popleft()

    def _network_loop(self) -> None:
        if not self.config.get('network_enable', True):
            return
        if random.random() < self.config.get('code_replication_prob', 0.1):
            self._code_replication()
        self._network_maintenance()

    # ------------------------------------------------------------------
    # Mutation & replication helpers
    # ------------------------------------------------------------------
    def _init_mutable_methods(self) -> None:
        method_names = [
            '_metabolism_loop',
            '_environment_loop',
            '_evolution_loop',
            '_memory_loop',
            '_network_loop',
        ]
        for name in method_names:
            if hasattr(self, name):
                self.mutable_methods.append(name)
                try:
                    source = inspect.getsource(getattr(self, name))
                    self._method_sources[name] = inspect.cleandoc(source)
                except (OSError, TypeError):
                    self._method_sources[name] = ''

    def _generate_quantum_dna(self) -> str:
        try:
            params = {
                'metabolism': random.uniform(0.5, 1.5),
                'mutation_rate': random.uniform(0.01, 0.1),
                'learning_rate': random.uniform(0.001, 0.01),
                'exploration': random.random(),
                'defense': random.uniform(0, 2),
                'pop_size': random.randint(4, 32),
                'crossover': random.uniform(0.1, 0.8),
                'op_bias': random.random(),
                'timeout': random.randint(400, 1600),
            }
            return self.genetic_encoder.encode(params)
        except Exception as exc:  # pragma: no cover - fallback
            logger.warning("Quantum DNA generation failed: %s", exc)
            return hashlib.sha3_512(os.urandom(64)).hexdigest()

    def _init_neural_architecture(self) -> Dict[str, Any]:
        return {
            'sensory_layers': [128, 64],
            'decision_layers': [64, 32, 16],
            'plasticity': 0.1,
        }

    def _code_replication(self) -> None:
        if not self.config.get('network_enable', True):
            return
        if self.energy < self.config['min_energy_for_replication']:
            return
        with self._lock:
            now = time.time()
            recent = [t for t in self._replication_times if now - t < 3600]
            if len(recent) >= self.config.get('max_replication_per_hour', 2):
                return
            self._replication_times = deque(recent, maxlen=100)
            self._replication_times.append(now)

        package = self._create_replication_package()
        targets = self._find_replication_targets()
        if not targets:
            return

        success = False
        for target in targets:
            if self.network_api.send_replication_package(target, package):
                success = True
                break
        if success:
            self.energy = max(0.0, self.energy - 10)
            self.pleasure = _clamp(self.pleasure + 0.1, 0.0, 1.0)
        else:
            self.stress = _clamp(self.stress + 0.05, 0.0, 1.0)

    def _create_replication_package(self) -> Dict[str, Any]:
        safe_config = {k: v for k, v in self.config.items() if k not in ('auth_token', 'allowlist')}
        package = {
            'metadata': {
                'source_node': self.node_id,
                'timestamp': time.time(),
                'code_version': self.code_version,
                'dna_fingerprint': self.dna[:32],
                'host': self.config.get('host'),
                'port': self.config.get('port'),
            },
            'core_code': {name: self._method_sources.get(name, '') for name in self.mutable_methods},
            'config': safe_config,
            'knowledge': self._json_sanitize(self.knowledge_base, max_depth=4),
            'dna_sequence': self.dna,
        }
        payload_bytes = json.dumps(package).encode('utf-8')
        digest = hashlib.sha256(payload_bytes).digest()
        signature = self._signing_key.sign(digest).hex()
        return {
            'payload': base64.b64encode(payload_bytes).decode('utf-8'),
            'sig': signature,
            'pubkey': self._pubkey_hex,
        }

    def _find_replication_targets(self) -> List[str]:
        try:
            active_nodes = self.blockchain.get_active_nodes()
            addr_map = self.blockchain.get_node_address_map()
            candidates = [n for n in active_nodes if n != self.node_id and n in addr_map]
            random.shuffle(candidates)
            return candidates[: min(3, len(candidates))]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Knowledge helpers
    # ------------------------------------------------------------------
    def _json_sanitize(self, obj: Any, max_depth: int = 4) -> Any:
        return json_sanitize(obj, max_depth=max_depth)

    def _prune_knowledge_base(self, max_items: int = 2000) -> None:
        if len(self.knowledge_base) <= max_items:
            return
        keys = list(self.knowledge_base.keys())
        drop = max(0, len(keys) - max_items)
        for key in keys[:drop]:
            self.knowledge_base.pop(key, None)

    # ------------------------------------------------------------------
    # Networking & integration
    # ------------------------------------------------------------------
    def _network_maintenance(self) -> None:
        try:
            addr_map = self.blockchain.get_node_address_map()
            for node_id, (host, _port, _pub) in list(addr_map.items())[:30]:
                if node_id == self.node_id:
                    continue
                if not is_allowed_destination(
                    host,
                    strict_check=self.config.get('strict_target_ip_check', True),
                    allow_local=self.config.get('local_discovery_enable', True),
                ):
                    continue
                try:
                    requests.get(f"http://{host}:{_port}/ping", timeout=1.0)
                except Exception:
                    continue
        except Exception:
            pass

    def _integrate_code(self, code_data: Dict[str, Any]) -> bool:
        if self.state == LifeState.REPLICATING:
            return False

        try:
            payload_b64 = code_data['payload']
            sig_hex = code_data['sig']
            sender_pubkey_hex = code_data.get('pubkey', '')

            payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
            digest = hashlib.sha256(payload_bytes).digest()
            vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sender_pubkey_hex))
            vk.verify(bytes.fromhex(sig_hex), digest)

            package = json.loads(payload_bytes.decode('utf-8'))
            metadata = package.get('metadata', {})
            source_node = metadata.get('source_node', '')

            addr_map = self.blockchain.get_node_address_map()
            if source_node not in addr_map:
                host = metadata.get('host')
                port = int(metadata.get('port', 0))
                if host and 1 <= port <= 65535:
                    if is_allowed_destination(
                        host,
                        strict_check=self.config.get('strict_target_ip_check', True),
                        allow_local=self.config.get('local_discovery_enable', True),
                    ):
                        self.blockchain.record_announce(source_node, host, port, sender_pubkey_hex)
                        addr_map = self.blockchain.get_node_address_map()

            if source_node not in addr_map:
                logger.error("Unknown source node %s", source_node)
                return False
            _, _, announced_pubkey = addr_map[source_node]
            if announced_pubkey and announced_pubkey != sender_pubkey_hex:
                logger.error("Pubkey mismatch for source %s", source_node)
                return False

            if not self._should_accept_code(package):
                return False

            with self._lock:
                self.state = LifeState.REPLICATING

            core_code = package.get('core_code', {})
            applied = False
            for method_name, code in core_code.items():
                if method_name not in self.mutable_methods:
                    continue
                old_code = self._method_sources.get(method_name, '')
                fitness = self.code_engine.evaluate_code_fitness(old_code, code)
                if fitness >= self.config.get('replication_threshold', 0.7):
                    if self.code_engine.hotswap_method(self, method_name, code):
                        self._method_sources[method_name] = code
                        self.code_version += 1
                        applied = True
                        self.blockchain.record_code_evolution(
                            self.node_id,
                            method_name,
                            old_code,
                            code,
                            {
                                'source': source_node,
                                'type': 'replication',
                                'code_version': self.code_version,
                                'applied': True,
                            },
                        )
            donor_dna = package.get('dna_sequence', self.dna)
            self.dna = self.genetic_encoder.recombine(self.dna, donor_dna)

            incoming_knowledge = package.get('knowledge', {})
            if isinstance(incoming_knowledge, dict):
                with self._kb_lock:
                    self.knowledge_base.update(incoming_knowledge)
                    self._prune_knowledge_base()

            return applied
        except Exception as exc:
            logger.error("Code integration failed: %s", exc)
            return False
        finally:
            with self._lock:
                self.state = LifeState.ACTIVE

    def _should_accept_code(self, package: Dict[str, Any]) -> bool:
        donor_version = package.get('metadata', {}).get('code_version', 1)
        ratio = donor_version / max(1, self.code_version)
        base = 0.4 + 0.3 * (ratio - 1)
        if self.energy < self.config.get('min_energy_for_replication', 50.0):
            base *= 0.5
        base = _clamp(base, 0.1, 0.9)
        if random.random() < self.config.get('quantum_mutation_prob', 0.05):
            perturbed = self.quantum_enhancer.generate_quantum_value(base)
            base = float(perturbed) if isinstance(perturbed, (int, float)) else base
        return random.random() < base

    # ------------------------------------------------------------------
    # Termination & utilities
    # ------------------------------------------------------------------
    def _terminate(self) -> None:
        if not self.is_alive:
            return
        self.is_alive = False
        self.state = LifeState.TERMINATED
        released_resources = {
            'energy': self.energy * 0.5,
            'knowledge': len(self.knowledge_base) * 0.1,
            'memory': len(self.short_term_memory) * 0.05 + len(self.long_term_memory) * 0.1,
        }
        try:
            self.environment.release_resources(released_resources)
            self.blockchain.record_death(
                self.node_id,
                {
                    'final_energy': self.energy,
                    'final_consciousness': self.consciousness_level,
                    'age': self.age,
                    'code_versions': self.code_engine.code_versions,
                    'released_resources': released_resources,
                },
            )
        except Exception:
            pass
        logger.critical("Life terminated: %s", self.node_id)

    def shutdown(self, wait: bool = False) -> None:
        self.is_alive = False
        if wait:
            for thread in self._threads:
                thread.join(timeout=2.0)
            if self.api_thread.is_alive():
                self.api_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Flask compatibility helpers
    # ------------------------------------------------------------------
    @property
    def api(self) -> Flask:
        return self.network_api.app

    def _assimilate(self, data: Dict[str, Any]) -> None:
        self._integrate_code(data)

    def _integrate_knowledge(self, knowledge: Dict[str, Any]) -> None:
        if isinstance(knowledge, dict):
            with self._kb_lock:
                self.knowledge_base.update(knowledge)
                self._prune_knowledge_base()

