"""
Digital environment simulator with real host sensing capabilities
"""

import random
import time
import copy
from collections import deque
from typing import Dict, List, Any

try:
    import psutil
except ImportError:
    psutil = None

from ..utils.common import _clamp
from ..imports import logger


class DigitalEnvironment:
    """Digital environment simulator with real host awareness"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.resources = {
            'cpu': random.randint(1, 100),
            'memory': random.randint(1, 100),
            'network': random.randint(1, 100),
            'quantum': random.randint(1, 100),
            'knowledge': 0,
            'energy': 0.0
        }
        self.threats: List[Dict[str, Any]] = []
        self._use_psutil = psutil is not None
        self._last_net = None
        self._last_scan_ts = None
        self._init_environment_model()

    def _init_environment_model(self):
        """Initialize environment prediction model"""
        self.env_history = deque(maxlen=100)
        self.resource_predictor = None

    def _sense_real(self):
        """Collect real host resources if psutil is available"""
        try:
            cpu = int(psutil.cpu_percent(interval=None))
            mem = int(psutil.virtual_memory().percent)
            now = time.time()
            net = 1
            io = psutil.net_io_counters()
            if self._last_net is None or self._last_scan_ts is None:
                self._last_net = io
                self._last_scan_ts = now
                net = 1
            else:
                dt = max(0.5, now - self._last_scan_ts)
                dbytes = (io.bytes_sent - self._last_net.bytes_sent) + (io.bytes_recv - self._last_net.bytes_recv)
                bps = dbytes / dt
                # Scale to 100% at 10MB/s
                net = int(_clamp((bps / 10_000_000.0) * 100.0, 1, 100))
                self._last_net = io
                self._last_scan_ts = now
            self.resources['cpu'] = int(_clamp(cpu, 1, 100))
            self.resources['memory'] = int(_clamp(mem, 1, 100))
            self.resources['network'] = int(_clamp(net, 1, 100))
            # Quantum channel drift
            self.resources['quantum'] = int(_clamp(self.resources['quantum'] + random.randint(-2, 2), 1, 100))
        except Exception:
            # Fallback to random walk
            for k in ('cpu', 'memory', 'network', 'quantum'):
                try:
                    self.resources[k] = int(_clamp(self.resources.get(k, 50) + random.randint(-3, 3), 1, 100))
                except Exception:
                    pass

    def scan(self):
        """Scan environment state"""
        if self._use_psutil:
            self._sense_real()
        else:
            for k in self.resources:
                try:
                    self.resources[k] += random.randint(-5, 5)
                    self.resources[k] = max(1, min(100, self.resources[k]))
                except Exception:
                    try:
                        self.resources[k] = max(0.0, min(100.0, float(self.resources[k]) + random.uniform(-2, 2)))
                    except Exception:
                        pass

        self.env_history.append(copy.deepcopy(self.resources))

        if random.random() < 0.1:
            self.threats.append({
                'type': random.choice(['virus', 'exploit', 'quantum_attack']),
                'severity': random.randint(1, 10),
                'ts': time.time()
            })

        if len(self.threats) > 50:
            self.threats = self.threats[-50:]

        return {
            'resources': self.resources,
            'threats': self.threats
        }

    def predict_resources(self, steps=5):
        """Predict future resource changes"""
        if len(self.env_history) < 10:
            return None
        try:
            recent = list(self.env_history)[-10:]
            avg_resources = {}
            for k in self.resources:
                try:
                    avg_resources[k] = sum(float(r[k]) for r in recent) / len(recent)
                except Exception:
                    avg_resources[k] = 0.0
            return {
                'predicted': avg_resources,
                'steps': steps,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            return None

    def release_resources(self, resources: Dict):
        """Release resources to environment"""
        for k, v in resources.items():
            if k in self.resources:
                try:
                    self.resources[k] = min(100, float(self.resources[k]) + float(v))
                except Exception:
                    pass