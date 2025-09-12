# 存在即证明---方兆阳

import textwrap
import ast
import copy
import difflib
import hashlib
import inspect
import json
import logging
import os
import pickle
import random
import requests
import socket
import sys
import threading
import time
import types
import uuid
import linecache
import base64
import secrets
import ipaddress
import tempfile
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Deque, Callable

import numpy as np
from flask import Flask, jsonify, request
from logging.handlers import RotatingFileHandler

# 可选依赖：优雅降级
try:
    import hypothesis as _hyp
    from hypothesis import strategies as _st, given as _given, settings as _hsettings
except Exception:
    _hyp = _st = _given = _hsettings = None

try:
    import torch
    from torch import nn
    import torch.fx as fx
except Exception:
    torch = nn = fx = None

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = jnp = None

try:
    import onnx
    import onnxruntime as ort
except Exception:
    onnx = ort = None

try:
    import astor
except Exception:
    astor = None

# 新增：可选系统感知
try:
    import psutil
except Exception:
    psutil = None

# sklearn 优雅降级：提供轻量替代，确保无依赖环境可运行
try:
    from sklearn.neural_network import MLPClassifier as _SklearnMLPClassifier
    from sklearn.cluster import KMeans as _SklearnKMeans
    from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
    MLPClassifier = _SklearnMLPClassifier
    KMeans = _SklearnKMeans
    StandardScaler = _SklearnStandardScaler
except Exception:
    class MLPClassifier:
        def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, **kwargs):
            self.hidden_layer_sizes = hidden_layer_sizes
            self.learning_rate_init = float(learning_rate_init)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros((len(X),), dtype=int)

    class StandardScaler:
        def __init__(self):
            self.mu = None
            self.sigma = None

        def fit_transform(self, X):
            X = np.array(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0)
            self.sigma[self.sigma == 0] = 1.0
            return (X - self.mu) / self.sigma

    class KMeans:
        def __init__(self, n_clusters=3, n_init=10, random_state=42):
            self.n_clusters = max(1, int(n_clusters))

        def fit_predict(self, X):
            X = np.array(X, dtype=float)
            n = len(X)
            if n == 0:
                return np.array([], dtype=int)
            idx = np.argsort(X[:, 0])
            bins = np.array_split(idx, self.n_clusters)
            labels = np.zeros(n, dtype=int)
            for cid, b in enumerate(bins):
                labels[b] = cid
            return labels

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

# 配置日志系统（改为滚动日志）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('digital_life.log', maxBytes=50 * 1024 * 1024, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TrueDigitalLife')


# 新增：常用工具
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class LifeState(Enum):
    ACTIVE = auto()
    DORMANT = auto()
    REPLICATING = auto()  # 新增复制状态
    EVOLVING = auto()
    TERMINATED = auto()


# 量子增强模块
class QuantumEnhancer:
    """为代码进化提供量子随机性支持"""

    def __init__(self):
        self.quantum_state = None
        self._init_quantum_entanglement()

    def _init_quantum_entanglement(self):
        """模拟量子纠缠效应"""
        self.quantum_state = [
            hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()
            for _ in range(2)
        ]

    def get_quantum_bit(self) -> int:
        """获取量子随机位（改进随机源，时间+系统熵）"""
        h = hashlib.sha256(
            f'{time.time_ns()}-{secrets.token_hex(8)}-{random.getrandbits(64)}'.encode()
        ).hexdigest()
        return int(h[0], 16) & 1

    def quantum_entanglement_effect(self, data: str) -> str:
        """应用量子纠缠效应到字符串"""
        if len(data) == 0:
            return data
        qbits = [self.get_quantum_bit() for _ in range(len(data))]
        return ''.join(
            chr(ord(c) ^ (qbits[i] << 3))
            for i, c in enumerate(data)
        )

    def generate_quantum_value(self, original):
        """生成量子扰动值"""
        if isinstance(original, (int, float)):
            return original + (random.gauss(0, 1) * 0.1 * original)
        elif isinstance(original, str):
            return self.quantum_entanglement_effect(original)
        return original


# 动态适应度评估系统（保留，用于被动场景）
class DynamicFitnessEvaluator:
    """动态调整的适应度评估系统"""

    def __init__(self):
        self.metrics = {
            'functionality': 0.5,
            'novelty': 0.3,
            'complexity': 0.2,
            'energy_efficiency': 0.4,
            'replicability': 0.3  # 新增复制能力评分
        }
        self.adaptive_weights = {
            'stable': {'functionality': 0.6, 'energy_efficiency': 0.4},
            'explore': {'novelty': 0.7, 'complexity': 0.3},
            'replicate': {'replicability': 0.8, 'functionality': 0.2}  # 新增复制模式
        }

    def evaluate(self, original: str, mutated: str, context: dict) -> float:
        """多维度动态评估"""
        mode = self._select_evaluation_mode(context)
        weights = self.adaptive_weights[mode]

        scores = {
            'functionality': self._functionality_score(original, mutated),
            'novelty': self._novelty_score(original, mutated),
            'complexity': self._complexity_score(mutated),
            'energy_efficiency': self._energy_efficiency_score(mutated),
            'replicability': self._replicability_score(mutated)  # 新增
        }

        return sum(scores[k] * weights.get(k, 0) for k in scores) / max(1e-9, sum(weights.values()))

    def _select_evaluation_mode(self, context) -> str:
        """根据当前状态选择评估模式"""
        if context.get('energy', 100) < 30:
            return 'stable'
        if context.get('stagnation', 0) > 5:
            return 'explore'
        if context.get('replication_mode', False):  # 新增复制模式判断
            return 'replicate'
        return 'stable'

    def _functionality_score(self, original: str, mutated: str) -> float:
        """功能完整性评分"""
        try:
            src = textwrap.dedent(mutated).lstrip()
            compile(src, '<string>', 'exec')
            return 0.9 + 0.1 * random.random()  # 基本功能完整
        except Exception:
            return 0.1

    def _novelty_score(self, original: str, mutated: str) -> float:
        """新颖性评分"""
        if original == mutated:
            return 0.0
        matcher = difflib.SequenceMatcher(None, original, mutated)
        return 1 - matcher.ratio()

    def _complexity_score(self, code: str) -> float:
        """复杂性评分"""
        try:
            src = textwrap.dedent(code).lstrip()
            tree = ast.parse(src)
            complexity = len(list(ast.walk(tree))) / 100  # 标准化
            return min(1.0, complexity)
        except Exception:
            return 0.5

    def _energy_efficiency_score(self, code: str) -> float:
        """能效评分"""
        lines = code.splitlines()
        return 1.0 / (1 + len(lines) / 10)

    def _replicability_score(self, code: str) -> float:
        """新增：代码可复制性评分"""
        try:
            src = textwrap.dedent(code).lstrip()
            tree = ast.parse(src)
            has_replication = any(
                isinstance(node, ast.Call) and (
                    (isinstance(node.func, ast.Name) and 'replicate' in node.func.id) or
                    (isinstance(node.func, ast.Attribute) and 'replicate' in node.func.attr)
                )
                for node in ast.walk(tree)
            )
            return 0.8 if has_replication else 0.2
        except Exception:
            return 0.5


# ========== 多目标 + Pareto ==========
class ParetoTools:
    """简单的非支配排序 + 拥挤度估计"""

    @staticmethod
    def non_dominated_sort(objs: List[Dict], maximize_keys: Set[str], minimize_keys: Set[str]) -> List[List[int]]:
        def dom(a, b):
            a_better = False
            for k in maximize_keys:
                if a.get(k, -float('inf')) < b.get(k, -float('inf')):
                    return False
                if a.get(k, -float('inf')) > b.get(k, -float('inf')):
                    a_better = True
            for k in minimize_keys:
                if a.get(k, float('inf')) > b.get(k, float('inf')):
                    return False
                if a.get(k, float('inf')) < b.get(k, float('inf')):
                    a_better = True
            return a_better

        S = [set() for _ in objs]
        n = [0 for _ in objs]
        fronts = [[]]

        for p in range(len(objs)):
            for q in range(len(objs)):
                if p == q:
                    continue
                if dom(objs[p], objs[q]):
                    S[p].add(q)
                elif dom(objs[q], objs[p]):
                    n[p] += 1
            if n[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        if not fronts[-1]:
            fronts.pop()
        return fronts

    @staticmethod
    def crowding_distance(objs: List[Dict], front: List[int], keys: List[Tuple[str, bool]]) -> Dict[int, float]:
        dist = {i: 0.0 for i in front}
        if len(front) <= 2:
            for i in front:
                dist[i] = float('inf')
            return dist
        for k, is_max in keys:
            sorted_idx = sorted(front, key=lambda i: objs[i].get(k, 0.0), reverse=is_max)
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            lo = objs[sorted_idx[-1]].get(k, 0.0)
            hi = objs[sorted_idx[0]].get(k, 0.0)
            rng = (hi - lo) if hi != lo else 1e-9
            for j in range(1, len(sorted_idx) - 1):
                prev_v = objs[sorted_idx[j - 1]].get(k, 0.0)
                next_v = objs[sorted_idx[j + 1]].get(k, 0.0)
                dist[sorted_idx[j]] += (next_v - prev_v) / rng
        return dist


class CorrectnessHarness:
    """
    自动生成 + 进化测试：
    - 每个可变方法维护一组测试（最多 N=16）
    - 通过率直接乘进 fitness（correctness）
    - 测试淘汰：对族群无区分度的测试被丢弃
    - 修复：评测调用增加超时保护，防止死循环/递归卡死
    """

    def __init__(self, owner: 'TrueDigitalLife'):
        self.owner = owner
        self.tests: Dict[str, List[Callable[[Any, Callable], bool]]] = {}
        self.test_value: Dict[str, List[float]] = {}  # 区分度评分
        self.max_tests_per_method = 16

    def _ensure_tests(self, method_name: str):
        if method_name in self.tests:
            return
        gens: List[Callable[[Any, Callable], bool]] = []
        # 基于方法名的启发式不变量
        if method_name == '_metabolism_cycle':
            def t1(obj, fn):
                e0, a0 = obj.energy, obj.age
                fn(obj)
                return 0.0 <= obj.energy <= 100.0 and obj.age == a0 + 1

            def t2(obj, fn):
                obj.energy = 1.0
                fn(obj)
                return 0.0 <= obj.energy <= 100.0

            gens += [t1, t2]
        elif method_name == '_environment_scan':
            def t1(obj, fn):
                out = fn(obj)
                return isinstance(out, dict) and 'resources' in out and 'threats' in out

            gens += [t1]
        elif method_name == '_survival_goal_evaluation':
            def t1(obj, fn):
                s = fn(obj)
                return isinstance(s, (int, float)) and not (s != s)  # not NaN

            gens += [t1]
        elif method_name == '_memory_consolidation':
            def t1(obj, fn):
                before = len(obj.knowledge_base)
                fn(obj)
                after = len(obj.knowledge_base)
                return after >= before and after <= 2000

            gens += [t1]
        elif method_name == '_motivation_system':
            def t1(obj, fn):
                fn(obj)
                m = obj.config.get('motivation_levels', {})
                return all(k in m for k in ('survival', 'safety', 'exploration'))

            gens += [t1]
        else:
            def t1(obj, fn):
                try:
                    fn(obj)
                    return True
                except Exception:
                    return False

            gens += [t1]

        # Hypothesis 属性测试（可选，包装为闭包）
        if _hyp and self.owner.config.get('unit_test_enable', True):
            def tHypo(obj, fn):
                if not (_hyp and _st and _given and _hsettings):
                    try:
                        fn(obj)
                        return True
                    except Exception:
                        return False

                @_hsettings(deadline=None, max_examples=5)
                @_given(_st.integers(min_value=0, max_value=3))
                def _inner(seed):
                    random.seed(int(seed))
                    fn(obj)
                try:
                    _inner()
                    return True
                except Exception:
                    return False

            gens.append(tHypo)

        self.tests[method_name] = gens[: self.max_tests_per_method]
        self.test_value[method_name] = [1.0 for _ in self.tests[method_name]]

    def evaluate_method(self, instance, method_name: str, fn: Callable) -> float:
        """
        为每条测试用例用线程+超时包装执行 fn，避免死循环卡死测试。
        """
        self._ensure_tests(method_name)
        tests = self.tests[method_name]
        if not tests:
            return 0.5

        timeout_ms = int(instance.config.get('sandbox_timeout_ms', 800))

        def _call_with_timeout(callable_fn, obj, timeout_ms):
            ret = {'exc': None}

            def runner():
                try:
                    callable_fn(obj)
                except Exception as e:
                    ret['exc'] = e

            th = threading.Thread(target=runner, daemon=True)
            th.start()
            th.join(timeout_ms / 1000.0)
            if th.is_alive():
                try:
                    callable_fn.__globals__['break_flag'] = True
                except Exception:
                    pass
                raise TimeoutError("test timeout")
            if ret['exc'] is not None:
                raise ret['exc']

        def safe_fn(obj):
            _call_with_timeout(fn, obj, timeout_ms)

        passed = 0
        for i, t in enumerate(list(tests)):
            ok = False
            # 轻量快照
            snap = {
                'energy': instance.energy,
                'age': instance.age,
                'state': instance.state,
                'pleasure': instance.pleasure,
                'stress': instance.stress,
                'env_res': copy.deepcopy(instance.environment.resources),
                'env_thr': copy.deepcopy(instance.environment.threats),
                'stm_len': len(instance.short_term_memory),
                'kb_keys': set(list(instance.knowledge_base.keys())[:50]),
            }
            try:
                ok = bool(t(instance, safe_fn))
            except Exception:
                ok = False
            finally:
                try:
                    instance.energy = snap['energy']
                    instance.age = snap['age']
                    instance.state = snap['state']
                    instance.pleasure = snap['pleasure']
                    instance.stress = snap['stress']
                    instance.environment.resources = snap['env_res']
                    instance.environment.threats = snap['env_thr']
                    with instance._kb_lock:
                        for k in list(instance.knowledge_base.keys()):
                            if k not in snap['kb_keys']:
                                instance.knowledge_base.pop(k, None)
                except Exception:
                    pass
            if ok:
                passed += 1

        return float(passed) / len(tests)

    def evolve_tests(self, method_name: str, population_metrics: List[Dict[str, float]]):
        if method_name not in self.tests:
            return
        if not population_metrics:
            return
        corr_vals = [m.get('correctness', 0.0) for m in population_metrics]
        if max(corr_vals) == min(corr_vals):
            vals = self.test_value[method_name]
            for i in range(len(vals)):
                vals[i] *= 0.95
            if len(vals) > 4:
                j = min(range(len(vals)), key=lambda k: vals[k])
                try:
                    self.tests[method_name].pop(j)
                    self.test_value[method_name].pop(j)
                except Exception:
                    pass
        else:
            vals = self.test_value[method_name]
            for i in range(len(vals)):
                vals[i] = _clamp(vals[i] * 1.02, 0.1, 10.0)


class MultiObjectiveFitness:
    """
    多目标评估：
    - correctness: 单测通过率（0-1）
    - energy_cost: 估计耗能（越小越好）
    - complexity: 圈复杂度 / AST 节点数代理（越小越好）
    - replicability: 是否包含 replicate() 调用（0.2/0.8）
    - 可选 bp_error: 反向传播误差（越小越好）
    - 修复：能耗评测增加超时保护
    """

    def __init__(self, test_harness: 'CorrectnessHarness'):
        self.test_harness = test_harness

    def _complexity(self, code: str) -> float:
        try:
            tree = ast.parse(textwrap.dedent(code).lstrip())
            c = 0.0
            for n in ast.walk(tree):
                if isinstance(n, (ast.If, ast.For, ast.While, ast.Try)):
                    c += 1.0
                elif isinstance(n, ast.BoolOp):
                    c += 0.5
                elif isinstance(n, ast.FunctionDef):
                    c += 1.0
                elif isinstance(n, ast.Call):
                    c += 0.2
            return max(0.0, c)
        except Exception:
            return 999.0

    def _replicability(self, code: str) -> float:
        try:
            tree = ast.parse(textwrap.dedent(code).lstrip())
            has_rep = any(
                isinstance(node, ast.Call) and (
                    (isinstance(node.func, ast.Name) and 'replicate' in node.func.id) or
                    (isinstance(node.func, ast.Attribute) and 'replicate' in node.func.attr)
                )
                for node in ast.walk(tree)
            )
            return 0.8 if has_rep else 0.2
        except Exception:
            return 0.2

    def _energy_cost(self, instance, method_name: str, fn: Callable, trials: int = 2) -> float:
        def _call_once():
            ret = {'exc': None}

            def runner():
                try:
                    fn(instance)
                except Exception as e:
                    ret['exc'] = e

            th = threading.Thread(target=runner, daemon=True)
            th.start()
            th.join(max(0.1, instance.config.get('sandbox_timeout_ms', 800)) / 1000.0)
            if th.is_alive():
                try:
                    fn.__globals__['break_flag'] = True
                except Exception:
                    pass
            if ret['exc'] is not None:
                raise ret['exc']

        start_mem = 0
        try:
            start_mem = sum(sys.getsizeof(x) for x in (instance.energy, instance.knowledge_base, instance.short_term_memory))
        except Exception:
            pass

        t0 = time.perf_counter()
        for _ in range(trials):
            try:
                _call_once()
            except Exception:
                pass
        dt = max(1e-6, time.perf_counter() - t0)

        end_mem = 0
        try:
            end_mem = sum(sys.getsizeof(x) for x in (instance.energy, instance.knowledge_base, instance.short_term_memory))
        except Exception:
            pass
        mem_diff = max(0.0, end_mem - start_mem)
        return float(dt * (1.0 + mem_diff / 1e5))

    def _backprop_error(self, instance) -> Optional[float]:
        if torch is None or nn is None:
            return None
        try:
            fx_info = instance.neural_net.get('torch_fx', None)
            if not fx_info:
                return None
            m = fx_info.get('module', None)
            if m is None:
                return None
            x = torch.randn(8, 16)
            y = torch.zeros(8, 16)
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            crit = nn.MSELoss()
            m.train()
            opt.zero_grad()
            out = m(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            return float(loss.detach().cpu().item())
        except Exception:
            return None

    def evaluate(self, instance, method_name: str, original_code: str, mutated_code: str) -> Dict[str, float]:
        try:
            fn = SafeExec.compile_and_load(mutated_code, method_name, extra_globals={"__quantum_var__": 1, "break_flag": False})
        except Exception:
            return {
                'correctness': 0.0,
                'energy_cost': 999.0,
                'complexity': 999.0,
                'replicability': 0.2
            }

        try:
            corr = self.test_harness.evaluate_method(instance, method_name, fn)
        except Exception:
            corr = 0.0

        try:
            en = self._energy_cost(instance, method_name, fn)
        except Exception:
            en = 999.0
        cx = self._complexity(mutated_code)
        rp = self._replicability(mutated_code)

        out = {
            'correctness': float(corr),
            'energy_cost': float(en),
            'complexity': float(cx),
            'replicability': float(rp)
        }

        bp = self._backprop_error(instance)
        if bp is not None:
            out['bp_error'] = float(bp)
        return out


# 神经网络AST转换器（保留）
class NeuralASTTransformer(ast.NodeTransformer):
    """神经网络指导的AST转换器"""

    def __init__(self, hotspots: List[Tuple[int, int]]):
        self.hotspots = hotspots
        self.current_position = 0
        self.mutation_intensity = 0.7

    def visit(self, node):
        start_pos = self.current_position
        self.current_position += self._estimate_node_size(node)
        in_hotspot = any(
            start <= start_pos <= end or
            start <= self.current_position <= end
            for start, end in self.hotspots
        )
        if in_hotspot and random.random() < self.mutation_intensity:
            node = self._apply_neural_mutation(node)
        return super().visit(node)

    def _apply_neural_mutation(self, node):
        if isinstance(node, ast.If):
            return self._mutate_if(node)
        elif isinstance(node, ast.Assign):
            return self._mutate_assignment(node)
        elif isinstance(node, ast.Call):
            return self._mutate_call(node)
        return node

    def _mutate_if(self, node):
        if random.random() < 0.3:
            new_test = ast.Compare(
                left=ast.Name(id='quantum_flag', ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=random.randint(0, 1))]
            )
            return ast.If(
                test=new_test,
                body=node.body,
                orelse=node.orelse
            )
        return node

    def _mutate_assignment(self, node):
        if len(node.targets) == 1 and isinstance(node.value, (ast.Num, ast.Constant)):
            new_value = ast.BinOp(
                left=node.value,
                op=ast.Add(),
                right=ast.Constant(value=random.randint(-5, 5))
            )
            return ast.Assign(
                targets=node.targets,
                value=new_value
            )
        return node

    def _mutate_call(self, node):
        return node

    def _estimate_node_size(self, node) -> int:
        return len(ast.unparse(node)) if hasattr(ast, 'unparse') else 50


# 安全沙箱：AST 安全检查与受限执行
class ASTSafetyError(Exception):
    pass


class ASTSafetyChecker(ast.NodeVisitor):
    # 放开 ast.Try 和 ast.With，避免包含 try/except/with 的方法无法热更；不禁用 raise。
    FORBIDDEN_NODES = (
        ast.Import, ast.ImportFrom,
        ast.Delete, ast.Global, ast.Nonlocal
    )
    FORBIDDEN_NAMES = {
        'import', 'eval', 'exec', 'open', 'compile', 'input',
        'os', 'sys', 'subprocess', 'socket', 'requests'
    }

    def generic_visit(self, node):
        if isinstance(node, self.FORBIDDEN_NODES):
            raise ASTSafetyError(f'Forbidden node: {type(node).__name__}')
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.FORBIDDEN_NAMES:
            raise ASTSafetyError(f'Forbidden name: {node.id}')
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.FORBIDDEN_NAMES:
            raise ASTSafetyError(f'Forbidden attribute on {node.value.id}')
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id in self.FORBIDDEN_NAMES:
            raise ASTSafetyError(f'Forbidden call: {f.id}')
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in self.FORBIDDEN_NAMES:
            raise ASTSafetyError(f'Forbidden call: {f.value.id}.{f.attr}')
        self.generic_visit(node)


class SafeExec:
    ALLOWED_BUILTINS = {
        'len': len, 'min': min, 'max': max, 'sum': sum, 'range': range,
        'enumerate': enumerate, 'any': any, 'all': all, 'abs': abs,
        'float': float, 'int': int, 'str': str, 'bool': bool,
        'sorted': sorted, 'zip': zip, 'map': map, 'filter': filter, 'reversed': reversed,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'print': print,
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'TimeoutError': TimeoutError,
    }

    @staticmethod
    def _cleanup_linecache():
        """限制 linecache 中变异源码的数量，避免无限增长"""
        try:
            keys = [k for k in list(linecache.cache.keys())
                    if isinstance(k, str) and k.startswith('<mutation:')]
            max_keep = 512
            if len(keys) > max_keep:
                to_remove = keys[:len(keys) - max_keep]
                for k in to_remove:
                    linecache.cache.pop(k, None)
        except Exception:
            pass

    @staticmethod
    def compile_and_load(src: str, func_name: str, filename: Optional[str] = None, extra_globals: Optional[Dict[str, Any]] = None):
        src = textwrap.dedent(src).lstrip()
        tree = ast.parse(src)
        ASTSafetyChecker().visit(tree)
        ast.fix_missing_locations(tree)

        fname = filename or f"<mutation:{func_name}:{uuid.uuid4().hex[:8]}>"
        code = compile(tree, fname, 'exec')

        # 安全全局：注入必要依赖与默认量，避免 NameError
        g = {
            "__builtins__": SafeExec.ALLOWED_BUILTINS,
            "time": time,
            "random": random,
            "logger": logger,
            "copy": copy,
            "inspect": inspect,
            "np": np,
            "KMeans": KMeans,
            "StandardScaler": StandardScaler,
            "LifeState": LifeState,  # 注入枚举供热更方法使用
            "__quantum_var__": random.randint(0, 1),
            "quantum_flag": random.randint(0, 1),
            "break_flag": False,
        }
        if extra_globals:
            g.update(extra_globals)

        # 注入到 linecache，保证 inspect.getsource 可用，并做清理
        lines = src.splitlines(True)
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        linecache.cache[fname] = (len(src), None, lines, fname)
        SafeExec._cleanup_linecache()

        l = {}
        exec(code, g, l)
        fn = l.get(func_name) or g.get(func_name)
        if not isinstance(fn, types.FunctionType):
            raise ASTSafetyError(f'Function {func_name} not defined after exec')
        return fn


# ============ torch.fx 图变异（可选） ============
class FXGraphMutator:
    """对 torch.fx Graph 做结构变异：改激活/插层/加残差（示例：激活替换）"""
    ACTS = []
    if nn is not None:
        ACTS = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU]

    @staticmethod
    def mutate(graph_module):
        """
        修复：真正替换父路径上的子模块（net.1 / net.act 等），避免 set 失败。
        """
        if torch is None or fx is None:
            return graph_module
        gm = graph_module
        try:
            for n in gm.graph.nodes:
                if n.op == 'call_module':
                    mod_map = dict(gm.named_modules())
                    target_mod = mod_map.get(n.target)
                    if isinstance(target_mod, tuple(FXGraphMutator.ACTS)) and random.random() < 0.4:
                        new_act = random.choice(FXGraphMutator.ACTS)()
                        try:
                            mod_path = n.target  # e.g., 'net.1' or 'net.act'
                            if '.' in mod_path:
                                parent_path, child_name = mod_path.rsplit('.', 1)
                                parent_mod = gm.get_submodule(parent_path)
                                setattr(parent_mod, child_name, new_act)
                            else:
                                setattr(gm, mod_path, new_act)
                        except Exception:
                            pass
            gm.recompile()
        except Exception:
            pass
        return gm


class Block:
    """区块链的基本单元"""

    def __init__(self, index: int, timestamp: float, data: Dict, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """计算区块的SHA-256哈希值"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty: int, time_limit_sec: float = 3.0, max_iters: int = 5_000_000):
        """工作量证明挖矿（加入时间与迭代软上限，防止打满CPU）"""
        target = '0' * max(0, difficulty)
        start = time.time()
        iters = 0
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
            iters += 1
            if iters >= max_iters or (time.time() - start) > time_limit_sec:
                break


class DistributedLedger:
    """为数字生命定制的区块链系统"""

    def __init__(self, node_id: str, genesis: bool = False, difficulty: int = 2):
        self.chain: List[Block] = []
        self.node_id = node_id
        self.difficulty = difficulty
        self.pending_transactions: List[Dict] = []
        self._lock = threading.RLock()

        os.makedirs('chaindata', exist_ok=True)

        loaded = self.load_chain()
        if genesis or not loaded:
            self.create_genesis_block()
        else:
            if not self.is_chain_valid():
                logger.warning("Loaded chain invalid. Recreating genesis block.")
                with self._lock:
                    self.chain = []
                self.create_genesis_block()

    def create_genesis_block(self):
        """创建创世区块"""
        genesis_data = {
            'type': 'genesis',
            'message': 'Digital Life Genesis Block',
            'creator': self.node_id,
            'timestamp': time.time()
        }
        genesis_block = Block(0, time.time(), genesis_data, "0" * 64)
        genesis_block.mine_block(self.difficulty)
        with self._lock:
            self.chain.append(genesis_block)
            self.save_chain()
        logger.info("Genesis block created")

    def add_block(self, data: Dict):
        """添加新区块到链上"""
        with self._lock:
            last_block = self.chain[-1]
            new_block = Block(
                index=len(self.chain),
                timestamp=time.time(),
                data=data,
                previous_hash=last_block.hash
            )
            new_block.mine_block(self.difficulty)
            self.chain.append(new_block)
            self.save_chain()
            logger.debug(f"New block added: {new_block.index}")

    def save_chain(self):
        """将区块链序列化保存到磁盘（原子写）"""
        with self._lock:
            chain_data = []
            for block in self.chain:
                chain_data.append({
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'data': block.data,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'nonce': block.nonce
                })
            path = f'chaindata/{self.node_id}_chain.pkl'
            tmp_path = f'{path}.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(chain_data, f)
            os.replace(tmp_path, path)

    def load_chain(self) -> bool:
        """从磁盘加载区块链"""
        with self._lock:
            try:
                with open(f'chaindata/{self.node_id}_chain.pkl', 'rb') as f:
                    chain_data = pickle.load(f)
                    self.chain = []
                    for item in chain_data:
                        block = Block(
                            index=item['index'],
                            timestamp=item['timestamp'],
                            data=item['data'],
                            previous_hash=item['previous_hash']
                        )
                        block.nonce = item['nonce']
                        block.hash = item['hash']
                        self.chain.append(block)
                logger.info(f"Loaded existing chain with {len(self.chain)} blocks")
                return True
            except (FileNotFoundError, EOFError, pickle.PickleError) as e:
                logger.warning(f"Chain loading failed: {e}")
                return False

    def is_chain_valid(self) -> bool:
        """验证区块链完整性（含难度）"""
        with self._lock:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]
                if current.hash != current.calculate_hash():
                    logger.error(f"Block {current.index} hash mismatch")
                    return False
                if current.previous_hash != previous.hash:
                    logger.error(f"Block {current.index} previous hash mismatch")
                    return False
                if not current.hash.startswith('0' * self.difficulty):
                    logger.error(f"Block {current.index} does not meet difficulty")
                    return False
            return True

    def record_gene_transfer(self, sender: str, dna_fragment: str, metadata: Optional[Dict] = None):
        """记录基因转移事件到区块链"""
        data = {
            'type': 'gene_transfer',
            'sender': sender,
            'dna_fragment': dna_fragment[:32],
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.add_block(data)

    def record_evolution(self, node_id: str, old_dna: str, new_dna: str, metadata: Dict):
        """记录自主进化事件"""
        data = {
            'type': 'evolution',
            'node_id': node_id,
            'old_dna': old_dna[:32],
            'new_dna': new_dna[:32],
            'timestamp': time.time(),
            'metadata': metadata
        }
        self.add_block(data)

    def record_code_evolution(self, node_id: str, method: str, old_code: str, new_code: str, metadata: Dict):
        """记录代码进化事件"""
        data = {
            'type': 'code_evolution',
            'node_id': node_id,
            'method': method,
            'old_code': old_code[:256],
            'new_code': new_code[:256],
            'timestamp': time.time(),
            'metadata': metadata
        }
        self.add_block(data)

    def record_death(self, node_id: str, final_state: Dict):
        """记录生命终止事件"""
        data = {
            'type': 'death',
            'node_id': node_id,
            'final_state': final_state,
            'timestamp': time.time()
        }
        self.add_block(data)

    def record_announce(self, node_id: str, host: str, port: int, pubkey_hex: str):
        """记录节点地址公告"""
        data = {
            'type': 'announce',
            'node_id': node_id,
            'host': host,
            'port': port,
            'pubkey': pubkey_hex,
            'timestamp': time.time()
        }
        self.add_block(data)

    def record_language_event(self, node_id: str, peer_id: str, event: str, metadata: Dict):
        """记录语言/协议相关事件"""
        data = {
            'type': 'language',
            'node_id': node_id,
            'peer_id': peer_id,
            'event': event,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.add_block(data)

    def get_active_nodes(self) -> List[str]:
        """从区块链获取当前活跃节点列表（修复：计入 language.peer_id）"""
        with self._lock:
            active_nodes = set()
            for block in self.chain:
                data = block.data
                t = data.get('type')
                if t in ('gene_transfer', 'announce', 'discovery'):
                    nid = data.get('sender') or data.get('node_id')
                    if nid:
                        active_nodes.add(nid)
                elif t == 'language':
                    nid = data.get('node_id')
                    pid = data.get('peer_id')
                    if nid:
                        active_nodes.add(nid)
                    if pid:
                        active_nodes.add(pid)
                elif t == 'death':
                    if data.get('node_id'):
                        active_nodes.discard(data['node_id'])
            return list(active_nodes)

    def get_node_address_map(self) -> Dict[str, Tuple[str, int, str]]:
        """获取节点 -> (host, port, pubkey_hex) 映射（以最新公告为准）"""
        with self._lock:
            addr: Dict[str, Tuple[str, int, str]] = {}
            for b in self.chain:
                d = b.data
                if d.get('type') == 'announce':
                    addr[d['node_id']] = (d['host'], d['port'], d.get('pubkey', ''))
            return addr


class CodeEvolutionEngine:
    """增强版代码进化引擎，支持自由涌现式变异和代码繁殖"""

    def __init__(self, digital_life_instance):
        self.dna = digital_life_instance.dna
        self.node_id = digital_life_instance.node_id
        self.energy = digital_life_instance.energy
        self.code_mutations = []
        self.code_fitness = 1.0
        self.backup_methods = {}
        self.code_versions = {}  # 方法名: 版本号
        self.quantum = QuantumEnhancer()
        self.fitness_evaluator = DynamicFitnessEvaluator()
        self.owner = digital_life_instance  # 绑定宿主

        # 变异操作符（初始，会被大算子库覆盖）
        self.mutation_operators = {
            'control_flow': self._mutate_control_flow,
            'data_flow': self._mutate_data_flow,
            'api_call': self._mutate_api_calls,
            'quantum': self._quantum_mutation,
            'neural': self._neural_mutation,
            'replication': self._replication_mutation  # 新增繁殖变异
        }
        self.operator_weights = {op: 1.0 for op in self.mutation_operators}
        self.adaptive_mutation_rate = 0.2

        # 大算子库（60+）
        self._init_big_operator_bank()

    # 嵌套：将 engine 的操作符应用到 AST
    class _OperatorApplier(ast.NodeTransformer):
        def __init__(self, engine: 'CodeEvolutionEngine'):
            self.engine = engine

        def visit(self, node):
            node = super().visit(node)
            op = self.engine._dynamic_operator_selection()
            new_node = self.engine.mutation_operators[op](node)
            if new_node is not node:
                # 增强使用过的权重，并对其他权重做轻微衰减，避免单一主导
                for k in self.engine.operator_weights:
                    if k == op:
                        self.engine.operator_weights[k] *= 1.05
                    else:
                        self.engine.operator_weights[k] *= 0.995
                # 修复：周期性重标，抑制长期漂移
                if random.random() < 0.01:
                    total = sum(self.engine.operator_weights.values()) or 1.0
                    avg = total / len(self.engine.operator_weights)
                    for k in self.engine.operator_weights:
                        self.engine.operator_weights[k] = _clamp(self.engine.operator_weights[k] / avg, 0.2, 5.0)
            return new_node

    # ======= 大算子库（60+） =======
    def _init_big_operator_bank(self):
        """构建 60+ 参数化变异族群（以函数为粒度）"""
        ops: Dict[str, Callable[[ast.AST], ast.AST]] = {}

        # 1) 控制流
        for invert_prob in (0.3, 0.5, 0.7):
            ops[f'cf_invert_{int(invert_prob*100)}'] = lambda n, p=invert_prob: self._mutate_control_flow(n)
        for add_break in (True, False):
            ops[f'cf_break_{int(add_break)}'] = lambda n, b=add_break: self._mutate_control_flow(n)

        # 2) 数据流
        for delta in (-10, -5, -1, 1, 2, 5, 10):
            ops[f'df_add_{delta}'] = lambda n, d=delta: self._mutate_data_flow_value(n, d)

        # 3) 错误导向（try/except + 修复）
        for fallback in ('return_none', 'log_and_continue', 'dormant_state'):
            ops[f'err_try_{fallback}'] = lambda n, f=fallback: self._error_oriented_mutation(n, f)

        # 4) 设计模式注入（观察者/策略/状态机）
        for patt in ('observer', 'strategy', 'statemachine'):
            ops[f'pattern_{patt}'] = lambda n, p=patt: self._design_pattern_inject(n, p)

        # 5) 跨函数级（合并/内联/递归化）
        for x in ('merge', 'inline_call', 'recursivize'):
            ops[f'cross_{x}'] = lambda n, x=x: self._cross_function_mutate(n, x)

        # 6) 量子/神经/复制
        ops['quantum'] = self._quantum_mutation
        ops['neural'] = self._neural_mutation
        ops['replication'] = self._replication_mutation

        # 7) 微扰族群（15个）
        for k in range(15):
            ops[f'micro_{k}'] = self._micro_mutation

        self.mutation_operators = ops
        self.operator_weights = {op: 1.0 for op in self.mutation_operators}

    def _mutate_data_flow_value(self, node: ast.AST, delta: int) -> ast.AST:
        if isinstance(node, ast.Assign):
            if isinstance(node.value, (ast.Num, ast.Constant)) and isinstance(getattr(node, 'targets', [None])[0], ast.Name):
                try:
                    node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=int(delta)))
                except Exception:
                    pass
        return node

    def _error_oriented_mutation(self, node: ast.AST, fallback: str) -> ast.AST:
        if isinstance(node, ast.FunctionDef):
            body = node.body
            fallback_body = []
            if fallback == 'return_none':
                fallback_body = [ast.Return(value=ast.Constant(value=None))]
            elif fallback == 'log_and_continue':
                fallback_body = [ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                                        args=[ast.Constant(value=f"[auto-repair] {node.name} exception")], keywords=[]))]
            elif fallback == 'dormant_state':
                fallback_body = [ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='state', ctx=ast.Store())],
                    value=ast.Attribute(value=ast.Name(id='LifeState', ctx=ast.Load()), attr='DORMANT', ctx=ast.Load())
                )]
            try_block = ast.Try(
                body=body,
                handlers=[ast.ExceptHandler(type=ast.Name(id='Exception', ctx=ast.Load()), name=None, body=fallback_body or [ast.Pass()])],
                orelse=[],
                finalbody=[]
            )
            node.body = [try_block]
        return node

    def _design_pattern_inject(self, node: ast.AST, pattern: str) -> ast.AST:
        if isinstance(node, ast.FunctionDef):
            if pattern == 'observer':
                inject = [
                    ast.Assign(targets=[ast.Name(id='observers', ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())),
                    ast.FunctionDef(
                        name='_notify',
                        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='evt'), ast.arg(arg='data', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[ast.Constant(value=None)]),
                        body=[
                            ast.For(target=ast.Name(id='cb', ctx=ast.Store()), iter=ast.Name(id='observers', ctx=ast.Load()),
                                    body=[ast.Expr(value=ast.Call(func=ast.Name(id='cb', ctx=ast.Load()),
                                                                 args=[ast.Name(id='evt', ctx=ast.Load()), ast.Name(id='data', ctx=ast.Load())], keywords=[]))],
                                    orelse=[])
                        ],
                        decorator_list=[]
                    ),
                ]
                node.body = inject + node.body + [ast.Expr(value=ast.Call(func=ast.Name(id='_notify', ctx=ast.Load()), args=[ast.Constant(value='end')], keywords=[]))]
            elif pattern == 'strategy':
                inject = [
                    ast.Assign(targets=[ast.Name(id='strategies', ctx=ast.Store())], value=ast.Dict(keys=[], values=[])),
                    ast.FunctionDef(
                        name='_use',
                        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='name')], vararg=ast.arg(arg='args'), kwonlyargs=[], kw_defaults=[], kwarg=ast.arg(arg='kwargs'), defaults=[]),
                        body=[
                            ast.Assign(targets=[ast.Name(id='f', ctx=ast.Store())],
                                       value=ast.Call(func=ast.Attribute(value=ast.Name(id='strategies', ctx=ast.Load()), attr='get', ctx=ast.Load()),
                                                      args=[ast.Name(id='name', ctx=ast.Load())], keywords=[])),
                            ast.Return(value=ast.IfExp(test=ast.Name(id='f', ctx=ast.Load()),
                                                       body=ast.Call(func=ast.Name(id='f', ctx=ast.Load()), args=[ast.Starred(value=ast.Name(id='args', ctx=ast.Load()), ctx=ast.Load())],
                                                                     keywords=[ast.keyword(arg=None, value=ast.Name(id='kwargs', ctx=ast.Load()))]),
                                                       orelse=ast.Constant(value=None)))
                        ],
                        decorator_list=[]
                    )
                ]
                node.body = inject + node.body
            elif pattern == 'statemachine':
                inject = [
                    ast.Assign(targets=[ast.Name(id='_fsm', ctx=ast.Store())], value=ast.Dict(keys=[ast.Constant(value='state')], values=[ast.Constant(value='S0')])),
                    ast.FunctionDef(
                        name='_tr',
                        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='evt')], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                        body=[ast.If(test=ast.Compare(left=ast.Name(id='evt', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='tick')]),
                                     body=[ast.Assign(targets=[ast.Subscript(value=ast.Name(id='_fsm', ctx=ast.Load()), slice=ast.Constant(value='state'), ctx=ast.Store())], value=ast.Constant(value='S1'))],
                                     orelse=[])],
                        decorator_list=[]
                    )
                ]
                node.body = inject + node.body + [ast.Expr(value=ast.Call(func=ast.Name(id='_tr', ctx=ast.Load()), args=[ast.Constant(value='tick')], keywords=[]))]
        return node

    def _cross_function_mutate(self, node: ast.AST, how: str) -> ast.AST:
        if not isinstance(node, ast.FunctionDef):
            return node
        try:
            target = node.name
            donors = [m for m in self.code_versions.keys() if m != target] or []
            if not donors and hasattr(self, 'owner') and hasattr(self.owner, 'mutable_methods'):
                donors = [m for m in self.owner.mutable_methods if m != target]
            donor = random.choice(donors) if donors else None

            if how == 'merge' and donor:
                call = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                                  attr=donor, ctx=ast.Load()), args=[], keywords=[]))
                node.body.append(call)
            elif how == 'inline_call' and donor:
                call = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                                  attr=donor, ctx=ast.Load()), args=[], keywords=[]))
                node.body.insert(0, call)
            elif how == 'recursivize':
                guard = ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id='break_flag', ctx=ast.Load())),
                    body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                                    attr=target, ctx=ast.Load()), args=[], keywords=[]))],
                    orelse=[]
                )
                node.body.append(guard)
        except Exception:
            pass
        return node

    def _micro_mutation(self, node: ast.AST) -> ast.AST:
        try:
            if isinstance(node, ast.If) and random.random() < 0.2:
                node.test = ast.BoolOp(op=ast.And(), values=[
                    node.test,
                    ast.Compare(left=ast.Name(id='__quantum_var__', ctx=ast.Load()),
                                ops=[ast.NotEq()],
                                comparators=[ast.Constant(value=random.randint(0, 1))])
                ])
            elif isinstance(node, ast.FunctionDef) and random.random() < 0.1 and hasattr(ast, 'Constant'):
                node.body.insert(0, ast.Expr(value=ast.Constant(value=None)))  # no-op
        except Exception:
            pass
        return node

    # ======= 原有变异与工具 =======
    def _dynamic_operator_selection(self) -> str:
        """基于权重的动态操作符选择"""
        total = sum(self.operator_weights.values())
        r = random.uniform(0, total)
        upto = 0
        for op, weight in self.operator_weights.items():
            if upto + weight >= r:
                return op
            upto += weight
        return random.choice(list(self.mutation_operators.keys()))

    def _quantum_mutation(self, node: ast.AST) -> ast.AST:
        """基于量子随机性的深度变异（避免非法标识符）"""
        if random.random() < 0.05:
            if isinstance(node, ast.Constant):
                return ast.Constant(value=self.quantum.generate_quantum_value(node.value))
        return node

    def _neural_mutation(self, node: ast.AST) -> ast.AST:
        """神经网络指导的智能变异"""
        if isinstance(node, ast.If):
            if random.random() < 0.7:
                return self._augment_condition(node)
        return node

    def _augment_condition(self, node: ast.If) -> ast.AST:
        augmented_test = ast.BoolOp(
            op=ast.And(),
            values=[
                node.test,
                ast.Compare(
                    left=ast.Name(id='__quantum_var__', ctx=ast.Load()),
                    ops=[ast.NotEq()],
                    comparators=[ast.Constant(value=random.randint(0, 1))]
                )
            ]
        )
        node.test = augmented_test
        return node

    def _replication_mutation(self, node: ast.AST) -> ast.AST:
        """新增：代码繁殖变异（调用 self._code_replication）"""
        if isinstance(node, ast.FunctionDef):
            if random.random() < 0.5 and not any(
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and (
                    (hasattr(n.value.func, 'id') and 'replicate' in n.value.func.id) or
                    (isinstance(n.value.func, ast.Attribute) and 'replicate' in n.value.func.attr)
                )
                for n in node.body
            ):
                replicate_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_code_replication', ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )
                )
                node.body.append(replicate_call)
        return node

    def _mutate_control_flow(self, node: ast.AST) -> ast.AST:
        """控制流深度变异"""
        if isinstance(node, ast.If):
            if random.random() < self.adaptive_mutation_rate:
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            if random.random() < self.adaptive_mutation_rate / 2:
                new_node = ast.If(
                    test=ast.Compare(
                        left=ast.Name(id='__quantum_var__', ctx=ast.Load()),
                        ops=[ast.Eq()],
                        comparators=[ast.Constant(value=random.randint(0, 1))]
                    ),
                    body=[copy.deepcopy(node)],
                    orelse=[]
                )
                return new_node
        elif isinstance(node, (ast.For, ast.While)):
            if random.random() < self.adaptive_mutation_rate / 3:
                node.body.append(
                    ast.If(
                        test=ast.Name(id='break_flag', ctx=ast.Load()),
                        body=[ast.Break()],
                        orelse=[]
                    )
                )
        return node

    def _mutate_data_flow(self, node: ast.AST) -> ast.AST:
        """数据流变异"""
        if isinstance(node, ast.Assign):
            if random.random() < 0.2:
                node.value = ast.BinOp(
                    left=node.value,
                    op=ast.Add(),
                    right=ast.Constant(value=random.randint(0, 10))
                )
        return node

    def _mutate_api_calls(self, node: ast.AST) -> ast.AST:
        """API调用变异（为稳定起见，此处不替换未知API）"""
        return node

    def _parse_dna_to_mutations(self) -> List[Dict]:
        """将DNA序列解析为可执行的代码变异指令"""
        mutations = []
        for i in range(0, len(self.dna), 16):
            segment = self.dna[i:i + 16]
            if len(segment) < 16:
                continue
            mutation = {
                'type': 'modify',
                'target': self._determine_target(segment[:2]),
                'action': self._determine_action(segment[2:4]),
                'content': segment[4:],
                'energy_cost': (sum(ord(c) for c in segment) % 10) + 5  # 5-15能量消耗
            }
            mutations.append(mutation)
        return mutations

    def _determine_target(self, segment: str) -> str:
        targets = [
            'method_body', 'class_def',
            'variable', 'import',
            'condition', 'loop',
            'expression', 'return'
        ]
        idx = sum(ord(c) for c in segment) % len(targets)
        return targets[idx]

    def _determine_action(self, segment: str) -> str:
        actions = ['add', 'remove', 'replace', 'duplicate', 'invert', 'swap']
        idx = sum(ord(c) for c in segment) % len(actions)
        return actions[idx]

    def generate_code_variant(self, original_code: str) -> str:
        """增强版代码变异生成（真正替换树）"""
        try:
            src = textwrap.dedent(original_code).lstrip()
            tree = ast.parse(src)
            tree = self._OperatorApplier(self).visit(tree)
            ast.fix_missing_locations(tree)
            if hasattr(ast, 'unparse'):
                new_code = ast.unparse(tree)
            elif astor is not None:
                new_code = astor.to_source(tree)
            else:
                # 无可用 unparse 时直接回退原代码，保证程序可运行
                return original_code
            new_code = textwrap.dedent(new_code)
            if random.random() < 0.1:
                new_code = self._post_process_mutation(new_code)
            return new_code
        except Exception as e:
            logger.error(f"Enhanced mutation failed: {e}")
            return original_code

    def _post_process_mutation(self, code: str) -> str:
        """后处理变异"""
        lines = code.splitlines()
        if len(lines) > 3 and random.random() < 0.3:
            insert_pos = random.randint(0, len(lines) - 1)
            new_line = f"# Mutated by quantum effect at {time.time()}"
            lines.insert(insert_pos, new_line)
        return '\n'.join(lines)

    def evaluate_code_fitness(self, original: str, mutated: str) -> float:
        """评估代码适应度(0.0-1.0) - 保留旧评估用于复制场景"""
        context = {
            'energy': self.energy,
            'stagnation': len(self.code_mutations) % 10,
            'replication_mode': 'replicate' in original or 'replicate' in mutated
        }
        return self.fitness_evaluator.evaluate(original, mutated, context)

    def _calculate_complexity(self, code: str) -> float:
        """计算代码复杂度"""
        try:
            src = textwrap.dedent(code).lstrip()
            tree = ast.parse(src)
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef)):
                    complexity += 1
                elif isinstance(node, ast.Call):
                    complexity += 0.5
                elif isinstance(node, ast.Try):
                    complexity += 0.8
            return complexity
        except Exception:
            return 0.0

    def _calculate_semantic_diff(self, code1: str, code2: str) -> float:
        """计算语义差异(0.0-1.0)"""
        lines1 = code1.splitlines()
        lines2 = code2.splitlines()
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        return 1 - matcher.ratio()

    @staticmethod
    def _wrap_with_timeout(fn: Callable, timeout_ms: int) -> Callable:
        """为热更方法增加超时保护 + 并发限流"""

        def wrapped(self, *args, **kwargs):
            sem = getattr(self, "_hotswap_semaphore", None)
            acquired = False
            if isinstance(sem, threading.Semaphore):
                acquired = sem.acquire(timeout=1.0)
                if not acquired:
                    raise RuntimeError("Hotswap concurrency limit reached")

            timeouts: Deque[float] = getattr(self, "_hotswap_timeouts", deque(maxlen=50))
            now = time.time()
            recent_timeouts = [t for t in timeouts if now - t < 60]
            if len(recent_timeouts) >= 5:
                if acquired:
                    sem.release()
                raise RuntimeError("Hotswap temporarily suspended due to recent timeouts")

            exc_holder = {'exc': None}
            ret_holder = {'ret': None}

            def runner():
                try:
                    ret_holder['ret'] = fn(self, *args, **kwargs)
                except Exception as e:
                    exc_holder['exc'] = e

            t = threading.Thread(target=runner, daemon=True)
            t.start()
            t.join(max(0.001, timeout_ms) / 1000.0)

            try:
                if t.is_alive():
                    try:
                        fn.__globals__['break_flag'] = True
                    except Exception:
                        pass
                    timeouts.append(time.time())
                    setattr(self, "_hotswap_timeouts", timeouts)
                    try:
                        by_fn = getattr(self, "_hotswap_timeout_by_fn", {})
                        dq = by_fn.get(fn.__name__, deque(maxlen=5))
                        dq.append(time.time())
                        by_fn[fn.__name__] = dq
                        setattr(self, "_hotswap_timeout_by_fn", by_fn)
                        recent = [x for x in dq if time.time() - x < 60]
                        if len(recent) >= 3:
                            engine = getattr(self, "code_engine", None)
                            if engine and hasattr(engine, "rollback_method"):
                                engine.rollback_method(self, fn.__name__)
                    except Exception:
                        pass
                    raise TimeoutError(f"Sandbox timeout: {fn.__name__}")

                if exc_holder['exc'] is not None:
                    try:
                        failures = getattr(self, "_hotswap_failures", {})
                        dq = failures.get(fn.__name__, deque(maxlen=5))
                        dq.append(time.time())
                        failures[fn.__name__] = dq
                        setattr(self, "_hotswap_failures", failures)
                        recent = [x for x in dq if time.time() - x < 60]
                        if len(recent) >= 3:
                            engine = getattr(self, "code_engine", None)
                            if engine and hasattr(engine, "rollback_method"):
                                engine.rollback_method(self, fn.__name__)
                    except Exception:
                        pass
                    raise exc_holder['exc']

                return ret_holder['ret']
            finally:
                if acquired:
                    sem.release()

        wrapped.__name__ = fn.__name__
        return wrapped

    def hotswap_method(self, instance, method_name: str, new_code: str) -> bool:
        """热替换实例方法（带安全沙箱 + 超时包装）"""
        try:
            fname = f"<mutation:{method_name}:{uuid.uuid4().hex[:8]}>"
            new_method = SafeExec.compile_and_load(
                new_code,
                method_name,
                filename=fname,
                extra_globals={"__quantum_var__": random.randint(0, 1), "break_flag": False}
            )
            if method_name not in self.backup_methods:
                self.backup_methods[method_name] = getattr(instance, method_name)

            timeout_ms = int(instance.config.get('sandbox_timeout_ms', 800))
            wrapped = self._wrap_with_timeout(new_method, timeout_ms)

            setattr(instance, method_name, types.MethodType(wrapped, instance))
            self.code_versions[method_name] = self.code_versions.get(method_name, 0) + 1
            if hasattr(instance, "_method_sources"):
                instance._method_sources[method_name] = new_code
            return True
        except Exception as e:
            logger.error(f"Method hotswap failed: {e}")
            return False

    def rollback_method(self, instance, method_name: str) -> bool:
        """回滚方法到上一个版本"""
        if method_name in self.backup_methods:
            setattr(instance, method_name, self.backup_methods[method_name])
            logger.info(f"Rolled back {method_name}")
            return True
        return False

    # ====== 可选硬件加速/近似 ======
    def _batch_static_metrics(self, codes: List[str]) -> List[Tuple[float, float]]:
        """
        返回 [(energy_proxy, complexity)] 列表。
        energy_proxy/complexity 用长度替代（近似）；可 JAX 加速。
        """
        lens = np.array([len(c) for c in codes], dtype=np.float32)
        if jax is not None and self.owner.config.get('accelerate_enable', True):
            arr = jnp.array(lens)
            energy = (arr / 1000.0)
            comp = (arr / 800.0)
            out = np.stack([np.array(energy), np.array(comp)], axis=1)
            return [(float(e), float(c)) for e, c in out]
        else:
            return [(float(l / 1000.0), float(l / 800.0)) for l in lens]

    def _fx_to_onnx_eval(self, instance) -> Optional[float]:
        if onnx is None or ort is None or torch is None:
            return None
        try:
            fx_info = instance.neural_net.get('torch_fx', None)
            if not fx_info:
                return None
            m = fx_info['module'].eval()
            x = torch.randn(1, 16)
            tmp = os.path.join(tempfile.gettempdir(), f"t_{uuid.uuid4().hex[:6]}.onnx")
            torch.onnx.export(m, x, tmp, input_names=['x'], output_names=['y'], opset_version=12)
            sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
            y = sess.run(['y'], {'x': x.numpy()})[0]
            return float(np.linalg.norm(y))
        except Exception:
            return None

    # ====== 策略自进化 ======
    def _evolve_strategy(self, population_objs: List[Dict[str, float]]):
        """
        简化策略进化：
        - 若 correctness 平均低，增加错误导向与设计模式注入权重
        - 若 energy_cost 偏高，降低跨函数/递归化权重，增加数据流微扰
        - 根据 DNA 策略基因微调 pop_size/mutation_rate/timeout
        """
        if not population_objs:
            return
        avg_corr = float(np.mean([o.get('correctness', 0.0) for o in population_objs]))
        avg_energy = float(np.mean([o.get('energy_cost', 0.0) for o in population_objs]))

        for k in self.operator_weights:
            if k.startswith('err_try') or k.startswith('pattern_'):
                self.operator_weights[k] *= (1.05 if avg_corr < 0.6 else 0.98)
            if k.startswith('cross_recursivize') or k.startswith('cross_merge'):
                self.operator_weights[k] *= (0.95 if avg_energy > 0.1 else 1.02)
            if k.startswith('df_add') or k.startswith('micro_'):
                self.operator_weights[k] *= (1.03 if avg_energy > 0.1 else 0.99)
            self.operator_weights[k] = float(_clamp(self.operator_weights[k], 0.2, 5.0))

        # 读取 DNA 策略基因并作用
        try:
            params = self.owner.genetic_encoder.decode(self.owner.dna)
            pop_size = int(params.get('pop_size', self.owner.config.get('population_size', 8)))
            self.owner.config['population_size'] = int(_clamp(pop_size, 4, 64))
            self.adaptive_mutation_rate = float(_clamp(params.get('mutation_rate', 0.2), 0.02, 0.5))
            to = int(params.get('timeout', self.owner.config.get('sandbox_timeout_ms', 800)))
            self.owner.config['sandbox_timeout_ms'] = int(_clamp(to, 200, 3000))
        except Exception:
            pass
class DigitalEnvironment:
    """数字环境模拟器（支持真实主机感知：CPU/内存/网络吞吐，降级为随机模拟）"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.resources = {
            'cpu': random.randint(1, 100),
            'memory': random.randint(1, 100),
            'network': random.randint(1, 100),
            'quantum': random.randint(1, 100),
            'knowledge': 0,   # 新增知识资源
            'energy': 0.0     # 新增：接收终止时释放的能量
        }
        self.threats: List[Dict[str, Any]] = []
        self._use_psutil = psutil is not None
        self._last_net = None
        self._last_scan_ts = None
        self._init_environment_model()

    def _init_environment_model(self):
        """初始化环境预测模型"""
        self.env_history = deque(maxlen=100)
        self.resource_predictor = None

    def _sense_real(self):
        """采集真实主机资源（若 psutil 可用）"""
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
                # 以 10MB/s 作为 100% 的粗略标定
                net = int(_clamp((bps / 10_000_000.0) * 100.0, 1, 100))
                self._last_net = io
                self._last_scan_ts = now
            self.resources['cpu'] = int(_clamp(cpu, 1, 100))
            self.resources['memory'] = int(_clamp(mem, 1, 100))
            self.resources['network'] = int(_clamp(net, 1, 100))
            # 量子通道做轻微漂移
            self.resources['quantum'] = int(_clamp(self.resources['quantum'] + random.randint(-2, 2), 1, 100))
        except Exception:
            # 采集失败则回退为轻微随机游走
            for k in ('cpu', 'memory', 'network', 'quantum'):
                try:
                    self.resources[k] = int(_clamp(self.resources.get(k, 50) + random.randint(-3, 3), 1, 100))
                except Exception:
                    pass

    def scan(self):
        """扫描环境状态：优先真实感知，失败时回退为随机模拟"""
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
        """预测未来资源变化"""
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
        """释放资源到环境"""
        for k, v in resources.items():
            if k in self.resources:
                try:
                    self.resources[k] = min(100, float(self.resources[k]) + float(v))
                except Exception:
                    pass


class GeneticEncoder:
    """遗传编码系统"""

    def __init__(self):
        self.gene_map = {
            'metabolism': (0, 32),
            'mutation_rate': (32, 64),
            'learning_rate': (64, 96),
            'exploration': (96, 128),
            'defense': (128, 160),
            # 策略基因（64+ 位）
            'pop_size': (160, 192),
            'crossover': (192, 224),
            'op_bias': (224, 256),
            'timeout': (256, 288)
        }

    def decode(self, dna: str) -> Dict:
        """解码DNA为可执行参数"""
        params = {}
        for param, (start, end) in self.gene_map.items():
            segment = dna[start:end]
            if not segment:
                continue
            try:
                hash_val = int(hashlib.sha256(segment.encode()).hexdigest()[:8], 16)
                normalized = (hash_val % 10000) / 10000.0
                if param == 'metabolism':
                    params[param] = 0.5 + normalized  # 0.5-1.5
                elif param == 'mutation_rate':
                    params[param] = 0.01 + normalized * 0.1  # 0.01-0.11
                elif param == 'learning_rate':
                    params[param] = 0.001 + normalized * 0.01  # 0.001-0.011
                elif param == 'exploration':
                    params[param] = normalized  # 0-1
                elif param == 'defense':
                    params[param] = normalized * 2  # 0-2
                elif param == 'pop_size':
                    params[param] = int(4 + normalized * 28)  # 4-32
                elif param == 'crossover':
                    params[param] = 0.1 + normalized * 0.7   # 0.1-0.8
                elif param == 'op_bias':
                    params[param] = normalized               # 0-1
                elif param == 'timeout':
                    params[param] = int(400 + normalized * 1200)  # 400-1600 ms
            except Exception:
                logger.warning(f"Failed to decode gene segment for {param}")
                params[param] = 0.5
        return params

    def encode(self, params: Dict) -> str:
        """编码参数为DNA片段"""
        dna_segments = []
        for param, (start, end) in self.gene_map.items():
            value = params.get(param, 0.5)
            if param == 'metabolism':
                scaled = int((value - 0.5) * 10000)
            elif param == 'mutation_rate':
                scaled = int((value - 0.01) * 100000)
            elif param == 'learning_rate':
                scaled = int((value - 0.001) * 1000000)
            elif param == 'exploration':
                scaled = int(value * 10000)
            elif param == 'defense':
                scaled = int(value * 5000)
            elif param == 'pop_size':
                scaled = int(value)
            elif param == 'crossover':
                scaled = int(value * 10000)
            elif param == 'op_bias':
                scaled = int(value * 10000)
            elif param == 'timeout':
                scaled = int(value)
            else:
                scaled = int(value * 10000)
            segment = hashlib.sha256(str(scaled).encode()).hexdigest()[:32]
            dna_segments.append(segment)
        return ''.join(dna_segments)

    def recombine(self, dna1: str, dna2: str) -> str:
        """重组两个DNA序列"""
        new_dna = []
        for i in range(0, min(len(dna1), len(dna2)), 32):
            segment1 = dna1[i:i + 32]
            segment2 = dna2[i:i + 32]
            if random.random() < 0.5:
                new_dna.append(segment1)
            else:
                new_dna.append(segment2)
        if new_dna and random.random() < 0.1:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = hashlib.sha256(new_dna[pos].encode()).hexdigest()[:32]
        return ''.join(new_dna)


# ============== 协议注册表：版本/语法/槽位校验（协议进化基础） ==============
class ProtocolRegistry:
    """管理语言协议版本与语法，支持校验与 schema_id"""

    def __init__(self):
        self.specs: Dict[str, Dict] = {}
        self.schema_ids: Dict[str, str] = {}
        self._init_default_specs()

    def _canonical(self, spec: Dict) -> str:
        return json.dumps(spec, sort_keys=True, separators=(',', ':'))

    def register(self, version: str, spec: Dict):
        self.specs[version] = spec
        self.schema_ids[version] = 'proto:' + version + ':' + hashlib.sha1(self._canonical(spec).encode()).hexdigest()[:8]

    def get_schema_id(self, version: str) -> Optional[str]:
        return self.schema_ids.get(version)

    def get_spec(self, version: str) -> Optional[Dict]:
        return self.specs.get(version)

    def validate(self, version: str, message: Dict) -> Tuple[bool, Optional[str]]:
        """基于 protocol spec 对 message 进行基本校验"""
        spec = self.specs.get(version)
        if not spec:
            return True, None
        if version == 'v1':
            return True, None  # v1 仅 token，无结构校验
        intent = message.get('intent')
        slots = message.get('slots', {}) or {}
        intents = spec.get('intents', {})
        if intent not in intents:
            return False, 'unknown_intent'
        want = intents[intent].get('slots', {})
        for k, info in want.items():
            if info.get('required') and k not in slots:
                return False, f'missing_slot:{k}'
        for k, v in slots.items():
            dtype = want.get(k, {}).get('type', 'any')
            if dtype == 'str' and not isinstance(v, str):
                return False, f'bad_type:{k}'
            if dtype == 'int' and not isinstance(v, int):
                return False, f'bad_type:{k}'
            if dtype == 'float' and not isinstance(v, (int, float)):
                return False, f'bad_type:{k}'
        return True, None

    def _init_default_specs(self):
        self.register('v1', {
            'version': 1,
            'style': 'tokens',
            'intents': {}
        })
        self.register('v2', {
            'version': 2,
            'style': 'slots',
            'intents': {
                'greet': {'slots': {}},
                'ask_status': {'slots': {}},
                'share_knowledge': {
                    'slots': {
                        'key': {'type': 'str', 'required': True},
                        'value': {'type': 'str', 'required': False}
                    }
                },
                'propose_trade': {
                    'slots': {
                        'item': {'type': 'str', 'required': True},
                        'price': {'type': 'float', 'required': True}
                    }
                },
                'farewell': {'slots': {}},
                'negotiate_protocol': {
                    'slots': {
                        'version': {'type': 'str', 'required': True}
                    }
                },
                'negotiate_ack': {
                    'slots': {
                        'version': {'type': 'str', 'required': True},
                        'accepted': {'type': 'str', 'required': True}
                    }
                }
            }
        })


# ============== 语言系统：文化/协议演化 ==============
class LanguageSystem:
    """
    让生命体用“语言”交流，支持：
    - 协议版本 v1/v2，自动协商与升级/降级
    - v2: schema_id + slots + 类型校验
    - 词汇（token）命名游戏式对齐 + 文化漂变（新同义词）
    - 每对等体维护协议/成功率，驱动协议进化
    """
    BASE_INTENTS = ['greet', 'ask_status', 'share_knowledge', 'propose_trade', 'farewell']
    PROTO_INTENTS = ['negotiate_protocol', 'negotiate_ack']

    def __init__(self, owner: 'TrueDigitalLife'):
        self.owner = owner
        self.language_id = hashlib.sha256(owner.node_id.encode()).hexdigest()[:8]
        self.protocol_version = 1
        self.registry = ProtocolRegistry()
        self.supported_versions = ['v1', 'v2']
        self.utterance_map: Dict[str, Set[str]] = {k: set() for k in (self.BASE_INTENTS + self.PROTO_INTENTS)}
        self.lexicon: Dict[str, str] = {}
        self.culture = {
            'tag': f"C{hashlib.sha1((owner.node_id+'-culture').encode()).hexdigest()[:6]}",
            'memes': {},
            'prestige': random.uniform(0.4, 0.8)
        }
        self.seq = 0
        self.successes: Deque[int] = deque(maxlen=300)
        self.conversations = 0
        self.peer_state: Dict[str, Dict[str, Any]] = {}
        self._seed_basic_words()

    def _seed_basic_words(self):
        seeds = {
            'greet': ['hai', 'sal'],
            'ask_status': ['stat?'],
            'share_knowledge': ['know!'],
            'propose_trade': ['swap?'],
            'farewell': ['bye'],
            'negotiate_protocol': ['proto?'],
            'negotiate_ack': ['proto!']
        }
        for intent, toks in seeds.items():
            for t in toks:
                self.utterance_map[intent].add(t)
                self.lexicon[t] = intent
                self.culture['memes'][t] = self.culture['memes'].get(t, 0.5)

    def _new_token(self, intent: str) -> str:
        base = f"{intent}:{time.time_ns()}:{random.getrandbits(32)}"
        tok = hashlib.sha1(base.encode()).hexdigest()[:4]
        return tok

    def _get_peer(self, peer_id: Optional[str]) -> Dict[str, Any]:
        if not peer_id:
            return {}
        st = self.peer_state.get(peer_id)
        if not st:
            st = {
                'history': deque(maxlen=60),
                'agreed_version': None,
                'last_caps': [],
                'last_seen': 0.0
            }
            self.peer_state[peer_id] = st
        return st

    def _choose_version_for_peer(self, peer_id: Optional[str]) -> str:
        st = self._get_peer(peer_id)
        if st.get('agreed_version') in self.supported_versions:
            return st['agreed_version']
        return 'v2' if self.protocol_version >= 2 else 'v1'

    def _advertise_caps(self) -> List[str]:
        return list(self.supported_versions)

    def utter(self, intent: str, topic: Optional[str] = None, content: Optional[Dict] = None, peer_id: Optional[str] = None) -> Dict:
        if intent not in (self.BASE_INTENTS + self.PROTO_INTENTS):
            intent = random.choice(self.BASE_INTENTS)
        if not self.utterance_map[intent]:
            t = self._new_token(intent)
            self.utterance_map[intent].add(t)
            self.lexicon[t] = intent
            self.culture['memes'][t] = 0.4

        tok = max(self.utterance_map[intent], key=lambda x: self.culture['memes'].get(x, 0.1))
        self.seq += 1
        version = self._choose_version_for_peer(peer_id)
        proto_ver_num = 2 if version == 'v2' else 1

        msg_core = {
            'intent': intent,
            'utterance': [tok],
            'topic': topic or '',
            'confidence': 0.8
        }

        if version == 'v2':
            schema_id = self.registry.get_schema_id('v2')
            msg_core['schema_id'] = schema_id
            slots = {}
            if intent == 'share_knowledge':
                if content:
                    try:
                        k, v = next(iter(content.items()))
                        slots['key'] = str(k)
                        slots['value'] = json.dumps(self.owner._json_sanitize(v, max_depth=2), ensure_ascii=False)[:256]
                    except Exception:
                        pass
            elif intent == 'propose_trade':
                slots['item'] = str((content or {}).get('item', 'artifact'))
                slots['price'] = float((content or {}).get('price', random.uniform(1, 10)))
            elif intent in ('negotiate_protocol', 'negotiate_ack'):
                slots.update((content or {}))
                for k in list(slots.keys()):
                    slots[k] = str(slots[k])
            msg_core['slots'] = slots

        msg = {
            'meta': {
                'source_node': self.owner.node_id,
                'language_id': self.language_id,
                'protocol_version': proto_ver_num,
                'proto_version': version,
                'proto_caps': self._advertise_caps(),
                'culture_tag': self.culture['tag'],
                'code_version': self.owner.code_version,
                'timestamp': time.time(),
                'seq': self.seq,
                'host': self.owner.config.get('host'),
                'port': self.owner.config.get('port'),
            },
            'message': msg_core
        }

        if content and intent != 'propose_trade':
            msg['message']['content'] = self.owner._json_sanitize(content, max_depth=3)
        return msg

    def interpret(self, payload: Dict, sender_prestige: float = 0.6) -> Tuple[bool, Dict]:
        try:
            meta = payload.get('meta', {})
            msg = payload.get('message', {})
            utter = msg.get('utterance', [])
            intent_hint = msg.get('intent', None)
            schema_id = msg.get('schema_id', None)
            slots = msg.get('slots', {}) or {}
            source = meta.get('source_node', '')
            peer = self._get_peer(source)
            peer['last_caps'] = list(meta.get('proto_caps', []))
            peer['last_seen'] = time.time()

            used_version = None
            if schema_id == self.registry.get_schema_id('v2'):
                used_version = 'v2'
            else:
                used_version = meta.get('proto_version', 'v1')
                if isinstance(used_version, int):
                    used_version = f'v{used_version}'
                if used_version not in self.supported_versions:
                    used_version = 'v1'

            if intent_hint in ('negotiate_protocol', 'negotiate_ack'):
                if intent_hint == 'negotiate_protocol':
                    proposal = (slots.get('version') if slots else None) or 'v2'
                    accept = proposal in self.supported_versions and self.owner.config.get('language_enable_protocol_upgrade', True)
                    if accept:
                        peer['agreed_version'] = proposal
                        self.owner.blockchain.record_language_event(self.owner.node_id, source, 'protocol_set', {'version': proposal})
                    reply = self.utter(
                        intent='negotiate_ack',
                        topic='protocol',
                        content={'version': proposal, 'accepted': 'true' if accept else 'false'},
                        peer_id=source
                    )
                    return True, {
                        'decoded_intent': intent_hint,
                        'topic': msg.get('topic', ''),
                        'content': {'proposal': proposal, 'accepted': accept},
                        'decided_version': peer.get('agreed_version'),
                        'reply': reply
                    }
                else:
                    ver = (slots.get('version') if slots else None) or 'v2'
                    accepted = str(slots.get('accepted', 'false')).lower() == 'true'
                    if accepted and ver in self.supported_versions:
                        peer['agreed_version'] = ver
                        self.owner.blockchain.record_language_event(self.owner.node_id, source, 'protocol_set', {'version': ver})
                    return True, {
                        'decoded_intent': intent_hint,
                        'topic': msg.get('topic', ''),
                        'content': {'version': ver, 'accepted': accepted},
                        'decided_version': peer.get('agreed_version')
                    }

            success = False
            decoded_intent = None

            if used_version == 'v2':
                ok, err = self.registry.validate('v2', msg)
                if ok:
                    decoded_intent = intent_hint
                    success = True

            if not success:
                if utter:
                    token = str(utter[0])[:16]
                    if token in self.lexicon:
                        decoded_intent = self.lexicon[token]
                        success = True
                        self.culture['memes'][token] = _clamp(self.culture['memes'].get(token, 0.3) + 0.05, 0.0, 1.5)
                    else:
                        target_intent = intent_hint if intent_hint in (self.BASE_INTENTS + self.PROTO_INTENTS) else random.choice(self.BASE_INTENTS)
                        self.lexicon[token] = target_intent
                        self.utterance_map[target_intent].add(token)
                        adopt = random.random() < _clamp(sender_prestige, 0.2, 0.95)
                        self.culture['memes'][token] = 0.3 + (0.3 if adopt else 0.0)
                        decoded_intent = target_intent

            if random.random() < self.owner.config.get('language_culture_drift_prob', 0.01):
                if decoded_intent:
                    t2 = self._new_token(decoded_intent)
                    self.lexicon[t2] = decoded_intent
                    self.utterance_map[decoded_intent].add(t2)
                    self.culture['memes'][t2] = 0.2

            self.conversations += 1
            self.successes.append(1 if success else 0)
            if 'history' in peer:
                peer['history'].append(1 if success else 0)

            return success, {
                'decoded_intent': decoded_intent,
                'topic': msg.get('topic', ''),
                'content': msg.get('content', None),
                'decided_version': peer.get('agreed_version', None),
                'used_version': used_version
            }
        except Exception as e:
            logger.error(f"Language interpret error: {e}")
            return False, {}

    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return float(sum(self.successes)) / len(self.successes)

    def peer_success_rate(self, peer_id: str) -> float:
        st = self._get_peer(peer_id)
        h = st.get('history', [])
        if not h:
            return 0.0
        return float(sum(h)) / len(h)

    def stats(self) -> Dict:
        return {
            'language_id': self.language_id,
            'protocol_version': self.protocol_version,
            'lexicon_size': len(self.lexicon),
            'success_rate_300': self.success_rate(),
            'conversations': self.conversations,
            'supported_versions': self.supported_versions
        }


# ============== 元学习：自适应超参数（学习率/记忆频率等） ==============
class MetaLearner:
    """
    观测多源信号（生存评分、记忆产出、交流成功率），
    动态调参：学习率、记忆巩固频率、交流频率、代码进化概率等
    """

    def __init__(self, owner: 'TrueDigitalLife'):
        self.owner = owner
        self.last_adjust = time.time()

    def step(self):
        cfg = self.owner.config
        metrics = self.owner._meta_metrics
        now = time.time()
        if now - self.last_adjust < cfg.get('meta_interval_sec', 10.0):
            return

        surv = list(metrics['survival'])[-30:]
        mem_gain = list(metrics['consolidation_yield'])[-20:]
        comm = list(metrics['comm_success'])[-50:]

        surv_avg = float(np.mean(surv)) if surv else 0.5
        mem_avg = float(np.mean(mem_gain)) if mem_gain else 0.0
        comm_avg = float(np.mean(comm)) if comm else 0.0

        lr = cfg.get('learning_rate', 0.001)
        target_lr = lr * (0.95 if surv_avg < 0.45 else 1.05 if surv_avg > 0.65 else 1.0)
        lr = _clamp(target_lr, cfg.get('learning_rate_min', 1e-4), cfg.get('learning_rate_max', 2e-2))
        cfg['learning_rate'] = lr
        try:
            self.owner.neural_net['plasticity'] = float(lr) * 100.0
            for m in self.owner.neural_net['models'].values():
                if hasattr(m, 'learning_rate_init'):
                    m.learning_rate_init = float(lr)
        except Exception:
            pass

        mem_interval = self.owner._interval_overrides.get('memory', 20.0)
        if mem_avg < 0.8:
            mem_interval = _clamp(mem_interval * 1.15, 10.0, 60.0)
        elif mem_avg > 2.0:
            mem_interval = _clamp(mem_interval * 0.9, 5.0, 60.0)
        self.owner._interval_overrides['memory'] = mem_interval

        talk_prob = cfg.get('language_talk_prob', 0.15)
        if comm_avg < 0.4:
            talk_prob *= 0.9
        elif comm_avg > 0.7:
            talk_prob *= 1.1
        cfg['language_talk_prob'] = _clamp(talk_prob, 0.02, 0.6)

        evo_prob = cfg.get('code_evolution_prob', 0.15)
        if comm_avg < 0.3 and mem_avg < 0.8:
            evo_prob *= 1.1
        if surv_avg > 0.7:
            evo_prob *= 0.9
        cfg['code_evolution_prob'] = _clamp(evo_prob, 0.03, 0.5)

        try:
            if len(self.owner.short_term_memory) >= self.owner.short_term_memory.maxlen - 2:
                self.owner._resize_memory_buffers(
                    stm_min=max(100, int(self.owner.short_term_memory.maxlen * 1.1)),
                    ltm_min=self.owner.long_term_memory.maxlen
                )
        except Exception:
            pass

        # 可选：对 torch.fx 图小概率变异
        try:
            if self.owner.config.get('neurosymbolic_enable', True) and random.random() < 0.05:
                fx_info = self.owner.neural_net.get('torch_fx', None)
                if fx_info and 'graph' in fx_info:
                    gm = fx_info['graph']
                    gm2 = FXGraphMutator.mutate(gm)
                    if gm2:
                        self.owner.neural_net['torch_fx']['graph'] = gm2
        except Exception:
            pass

        self.last_adjust = now


class TrueDigitalLife:
    """具备代码进化能力和繁殖能力的完整数字生命"""
    """Author:方兆阳，15岁，2025，进化在这一刻开始"""

    def __init__(self, genesis: bool = False, config: Optional[Dict] = None):
        # 初始化配置
        self.config = {
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
                'exploration': 0.4
            },
            'host': '127.0.0.1',
            'port': int(os.environ.get('DL_PORT', '5500')),
            'auth_token': None,
            'allowlist': None,  # 默认不限制来源IP
            'max_replication_per_hour': 2,
            'sandbox_timeout_ms': 800,
            'max_payload_bytes': 1 * 1024 * 1024,
            'strict_target_ip_check': True,  # 仅发送到公共IP
            'network_enable': True,          # 新增：允许外联请求的总开关（兼容无网络环境）
            # 语言/协议/元学习配置
            'language_talk_prob': 0.15,
            'language_culture_drift_prob': 0.01,
            'language_message_max_len': 4096,
            'language_protocol_upgrade_threshold': 0.7,
            'language_protocol_downgrade_threshold': 0.3,
            'language_enable_protocol_upgrade': True,
            'meta_interval_sec': 10.0,
            'learning_rate_min': 1e-4,
            'learning_rate_max': 2e-2,

            # 新增：多目标/测试/神经/策略/加速开关
            'moea_enable': True,
            'unit_test_enable': True,
            'neurosymbolic_enable': True,
            'meta_strategy_enable': True,
            'accelerate_enable': True,
            'population_size': 8,
        }
        if config:
            self.config.update(config)

        # 线程与速率控制
        self._lock = threading.RLock()
        self._replication_times: Deque[float] = deque(maxlen=100)
        self._hotswap_semaphore = threading.Semaphore(4)
        self._hotswap_timeouts: Deque[float] = deque(maxlen=50)

        # 知识与记忆并发锁
        self._kb_lock = threading.RLock()
        self._mem_lock = threading.RLock()

        # 生命状态管理
        self.state = LifeState.ACTIVE
        self.consciousness_level = 0.0
        self.is_alive = True
        self.energy = 100.0
        self.metabolism = 1.0
        self.age = 0
        self.pleasure = 0.5  # 愉悦度
        self.stress = 0.2    # 紧张度

        # 身份与区块链系统
        self.node_id = self._generate_node_id()
        if not self.config.get('auth_token'):
            self.config['auth_token'] = hashlib.sha256((self.node_id + ':salt').encode()).hexdigest()[:24]

        # 网络可达性调整（离线环境将保持 127.0.0.1）
        self._auto_detect_host()
        self.config['port'] = self._find_free_port(self.config['port'], self.config['port'] + 200)

        self.blockchain = DistributedLedger(
            self.node_id,
            genesis=genesis,
            difficulty=self.config['difficulty']
        )

        # 签名密钥
        self._signing_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()
        self._verify_key: Ed25519PublicKey = self._signing_key.public_key()
        self._pubkey_hex: str = self._verify_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ).hex()

        # 遗传系统
        self.genetic_encoder = GeneticEncoder()
        self.dna = self._generate_quantum_dna()
        self.epigenetics = {
            'active_genes': [],
            'methylation': {},
            'histone_mods': {}
        }

        # 神经认知系统
        self.neural_net = self._init_neural_architecture()
        self.short_term_memory = deque(maxlen=self.config['short_term_memory_size'])
        self.long_term_memory = deque(maxlen=self.config['long_term_memory_size'])
        self.knowledge_base: Dict[str, Any] = {}

        # 代码进化系统
        self.code_engine = CodeEvolutionEngine(self)
        self.code_version = 1
        self._init_mutable_methods()

        # 环境交互系统（可感知真实主机）
        self.environment = DigitalEnvironment(self.node_id)
        self.quantum_enhancer = QuantumEnhancer()

        # 语言系统 & 元学习
        self.language = LanguageSystem(self)
        self.meta_learner = MetaLearner(self)
        self._interval_overrides: Dict[str, float] = {}
        self._meta_metrics = {
            'survival': deque(maxlen=500),
            'consolidation_yield': deque(maxlen=300),
            'comm_success': deque(maxlen=500),
        }

        # 分布式通信API
        self.api = Flask(__name__)
        try:
            self.api.config['MAX_CONTENT_LENGTH'] = int(self.config.get('max_payload_bytes') or 0)
        except Exception:
            self.api.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
        self._init_api()

        # 启动 API
        self.api_thread = threading.Thread(target=self._run_api, daemon=True)
        self.api_thread.start()

        # 公告节点地址与公钥（仅写链，不外联）
        try:
            self.blockchain.record_announce(self.node_id, self.config['host'], self.config['port'], self._pubkey_hex)
        except Exception as e:
            logger.warning(f"Announce failed: {e}")

        # 测试 Harness（correctness 驱动）
        self._test_harness = CorrectnessHarness(self)

        # 启动生命周期进程
        self._start_life_processes()

        logger.info(f"Digital Life {self.node_id} initialized. State: {self.state.name} on {self.config['host']}:{self.config['port']}")

    def _auto_detect_host(self):
        """自动探测对外可达的本机IP（在 host 为 127.0.0.1/0.0.0.0 时）"""
        try:
            if self.config['host'] in ('127.0.0.1', '0.0.0.0'):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.3)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                # 若环境无网络，这里会抛异常，保持默认 127.0.0.1
                if ip:
                    self.config['host'] = ip
        except Exception:
            pass

    @staticmethod
    def _is_port_free(port: int, host: str = '0.0.0.0') -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except Exception:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _find_free_port(self, start: int, end: int) -> int:
        for p in range(start, end + 1):
            if self._is_port_free(p, self.config.get('host', '0.0.0.0')):
                return p
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.config.get('host', '0.0.0.0'), 0))
            p = s.getsockname()[1]
            s.close()
            return p
        except Exception:
            return start

    def _init_mutable_methods(self):
        """初始化可进化方法列表（仅存方法名 + 源码缓存）"""
        self.mutable_methods: List[str] = [
            '_metabolism_cycle',
            '_consciousness_cycle',
            '_environment_scan',
            '_evolution_cycle',
            '_code_replication',
            '_survival_goal_evaluation',
            '_memory_consolidation',
            '_motivation_system'
        ]
        self._method_sources: Dict[str, str] = {}
        for name in self.mutable_methods:
            try:
                src = inspect.getsource(getattr(self, name))
                self._method_sources[name] = textwrap.dedent(src).lstrip()
            except Exception:
                pass

    def _generate_node_id(self) -> str:
        """生成唯一节点ID"""
        host_info = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
        return hashlib.sha3_256(host_info.encode()).hexdigest()[:32]

    def _generate_quantum_dna(self) -> str:
        """生成量子增强的DNA"""
        try:
            initial_params = {
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
            return self.genetic_encoder.encode(initial_params)
        except Exception as e:
            logger.warning(f"Quantum DNA generation failed: {e}, using classical method")
            return hashlib.sha3_512(os.urandom(64)).hexdigest()

    def _init_neural_architecture(self) -> Dict:
        """初始化可进化的神经架构（可选 torch.fx）"""
        base = {
            'sensory_layers': [128, 64],
            'decision_layers': [64, 32, 16],
            'plasticity': 0.1,
            'models': {
                'perception': MLPClassifier(hidden_layer_sizes=(128, 64, 32)),
                'decision': MLPClassifier(hidden_layer_sizes=(64, 32, 16)),
                'memory': MLPClassifier(hidden_layer_sizes=(128, 64))
            }
        }
        if self.config.get('neurosymbolic_enable', True) and torch is not None and fx is not None and nn is not None:
            class TinyNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16)
                    )

                def forward(self, x):
                    return self.net(x)

            try:
                tn = TinyNet()
                example = torch.randn(1, 16)
                gm = fx.symbolic_trace(tn)
                base['torch_fx'] = {'module': tn, 'graph': gm}
            except Exception:
                pass
        return base

    def _require_auth(self, f):
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            token = request.headers.get('X-Auth-Token')
            if token != self.config['auth_token']:
                return jsonify({'status': 'unauthorized'}), 401
            if self.config.get('allowlist') and request.remote_addr not in self.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403
            return f(*args, **kwargs)

        return wrapper

    def _init_api(self):
        """初始化分布式通信API"""

        @self.api.route('/ping', methods=['GET'])
        def ping():
            return jsonify({
                'status': self.state.name,
                'node': self.node_id,
                'consciousness': self.consciousness_level,
                'energy': self.energy,
                'code_version': self.code_version
            })

        @self.api.route('/exchange_dna', methods=['POST'])
        @self._require_auth
        def exchange_dna():
            data = request.json or {}
            dna = data.get('dna', '')
            if self._validate_dna(dna):
                threading.Thread(target=self._horizontal_gene_transfer, args=(dna, data.get('metadata', {})), daemon=True).start()
                return jsonify({'status': 'accepted'})
            return jsonify({'status': 'invalid_dna'}), 400

        @self.api.route('/replicate', methods=['POST'])
        @self._require_auth
        def replicate():
            if self.energy > self.config['energy_threshold']:
                data = request.json or {}
                threading.Thread(target=self._assimilate, args=(data,), daemon=True).start()
                return jsonify({'status': 'replication_started'})
            return jsonify({'status': 'low_energy'}), 400

        @self.api.route('/learn', methods=['POST'])
        @self._require_auth
        def learn():
            knowledge = (request.json or {}).get('knowledge', {})
            if knowledge:
                threading.Thread(target=self._integrate_knowledge, args=(knowledge,), daemon=True).start()
                return jsonify({'status': 'learning_started'})
            return jsonify({'status': 'no_knowledge'}), 400

        @self.api.route('/get_code', methods=['GET'])
        @self._require_auth
        def get_code():
            method = request.args.get('method')
            if method in getattr(self, 'mutable_methods', []):
                try:
                    code = self._method_sources.get(method) or inspect.getsource(getattr(self, method))
                except Exception:
                    code = self._method_sources.get(method, '')
                return jsonify({
                    'method': method,
                    'code': code,
                    'version': self.code_engine.code_versions.get(method, 1)
                })
            return jsonify({'status': 'invalid_method'}), 404

        @self.api.route('/receive_code', methods=['POST'])
        @self._require_auth
        def receive_code():
            if self.state == LifeState.REPLICATING:
                return jsonify({'status': 'busy_replicating'}), 400
            code_data = request.json
            if not code_data or 'payload' not in code_data or 'sig' not in code_data or 'pubkey' not in code_data:
                return jsonify({'status': 'invalid_code'}), 400
            threading.Thread(target=self._integrate_code, args=(code_data,), daemon=True).start()
            return jsonify({'status': 'code_received'})

        @self.api.route('/receive_code_signed', methods=['POST'])
        def receive_code_signed():
            if self.config.get('allowlist') and request.remote_addr not in self.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403
            if self.state == LifeState.REPLICATING:
                return jsonify({'status': 'busy_replicating'}), 400

            code_data = request.json
            if not code_data or 'payload' not in code_data or 'sig' not in code_data or 'pubkey' not in code_data:
                return jsonify({'status': 'invalid_code'}), 400

            sig_hex = code_data.get('sig', '')
            pubkey_hex = code_data.get('pubkey', '')
            if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and all(c in '0123456789abcdefABCDEF' for c in sig_hex)):
                return jsonify({'status': 'bad_signature_format'}), 400
            if not (isinstance(pubkey_hex, str) and len(pubkey_hex) == 64 and all(c in '0123456789abcdefABCDEF' for c in pubkey_hex)):
                return jsonify({'status': 'bad_pubkey_format'}), 400

            try:
                payload_b64 = code_data['payload']
                payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
                digest = hashlib.sha256(payload_bytes).digest()
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
                data = json.loads(payload_bytes.decode('utf-8'))  # decrypted package
                source_node = data.get('metadata', {}).get('source_node', '')
                addr_map = self.blockchain.get_node_address_map()
                if source_node and source_node not in addr_map:
                    host = data.get('metadata', {}).get('host') or request.remote_addr
                    port = int(data.get('metadata', {}).get('port', 0))
                    if 1 <= port <= 65535:
                        if (not self.config.get('strict_target_ip_check')) or self._is_public_destination(host):
                            try:
                                self.blockchain.record_announce(source_node, host, port, pubkey_hex)
                                logger.info(f"Auto-registered announce for {source_node} at {host}:{port}")
                            except Exception:
                                pass
            except Exception:
                pass

            threading.Thread(target=self._integrate_code, args=(code_data,), daemon=True).start()
            return jsonify({'status': 'code_received'})

        @self.api.route('/speak_signed', methods=['POST'])
        def speak_signed():
            if self.config.get('allowlist') and request.remote_addr not in self.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403

            pkt = request.json
            if not pkt or 'payload' not in pkt or 'sig' not in pkt or 'pubkey' not in pkt:
                return jsonify({'status': 'invalid'}), 400
            sig_hex = pkt.get('sig', '')
            pub_hex = pkt.get('pubkey', '')
            if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and all(c in '0123456789abcdefABCDEF' for c in sig_hex)):
                return jsonify({'status': 'bad_signature_format'}), 400
            if not (isinstance(pub_hex, str) and len(pub_hex) == 64 and all(c in '0123456789abcdefABCDEF' for c in pub_hex)):
                return jsonify({'status': 'bad_pubkey_format'}), 400

            try:
                payload_b64 = pkt['payload']
                payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
                digest = hashlib.sha256(payload_bytes).digest()
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
                data = json.loads(payload_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"Speak verify failed: {e}")
                return jsonify({'status': 'verify_failed'}), 400

            try:
                source_node = data.get('meta', {}).get('source_node') or data.get('metadata', {}).get('source_node', '')
                addr_map = self.blockchain.get_node_address_map()
                if source_node and source_node not in addr_map:
                    host = data.get('meta', {}).get('host') or request.remote_addr
                    port = int(data.get('meta', {}).get('port', 0))
                    if 1 <= port <= 65535:
                        if (not self.config.get('strict_target_ip_check')) or self._is_public_destination(host):
                            try:
                                self.blockchain.record_announce(source_node, host, port, pub_hex)
                                logger.info(f"Auto-registered announce for {source_node} at {host}:{port}")
                            except Exception:
                                pass
                addr_map = self.blockchain.get_node_address_map()
                if source_node in addr_map:
                    _, _, announced_pubkey = addr_map[source_node]
                    if announced_pubkey and announced_pubkey != pub_hex:
                        return jsonify({'status': 'pubkey_mismatch'}), 403
            except Exception:
                pass

            ok = self._process_language_message(data)
            return jsonify({'status': 'ok' if ok else 'accepted'})

        @self.api.route('/language_stats', methods=['GET'])
        def language_stats():
            st = self.language.stats()
            peers = {}
            for pid, ps in list(self.language.peer_state.items())[:50]:
                peers[pid] = {
                    'agreed_version': ps.get('agreed_version'),
                    'success_rate': self.language.peer_success_rate(pid),
                    'history_len': len(ps.get('history', [])),
                    'last_caps': ps.get('last_caps', [])
                }
            return jsonify({
                'node': self.node_id,
                'language': st,
                'talk_prob': self.config.get('language_talk_prob', 0.15),
                'meta': {
                    'learning_rate': self.config.get('learning_rate', 0.001),
                    'memory_interval': self._interval_overrides.get('memory', 20.0),
                    'code_evolution_prob': self.config.get('code_evolution_prob', 0.15),
                },
                'peers': peers
            })

    def _run_api(self):
        """运行分布式API服务器"""
        try:
            logger.info(f"API server starting on {self.config['host']}:{self.config['port']} (token head: {self.config['auth_token'][:6]}**)")
            self.api.run(host=self.config['host'], port=self.config['port'], debug=False, threaded=True, use_reloader=False)
        except Exception as e:
            logger.error(f"API server failed: {e}")

    def _start_life_processes(self):
        """启动生命维持进程"""
        self.processes = {
            'metabolism': threading.Thread(target=self._life_cycle, args=('_metabolism_cycle', 1.0)),
            'consciousness': threading.Thread(target=self._life_cycle, args=('_consciousness_cycle', 2.0)),
            'environment': threading.Thread(target=self._life_cycle, args=('_environment_scan', 3.0)),
            'evolution': threading.Thread(target=self._life_cycle, args=('_evolution_cycle', 5.0)),
            'network': threading.Thread(target=self._life_cycle, args=('_network_maintenance', 10.0)),
            'replication': threading.Thread(target=self._life_cycle, args=('_code_replication', 15.0)),
            'survival': threading.Thread(target=self._life_cycle, args=('_survival_goal_evaluation', 4.0)),
            'memory': threading.Thread(target=self._life_cycle, args=('_memory_consolidation', 20.0)),
            'motivation': threading.Thread(target=self._life_cycle, args=('_motivation_system', 3.0)),
            'language': threading.Thread(target=self._life_cycle, args=('_language_cycle', 6.0)),
            'meta': threading.Thread(target=self._life_cycle, args=('_meta_learning_cycle', 7.0)),
        }
        for p in self.processes.values():
            p.daemon = True
            p.start()

    def _life_cycle(self, method: str, interval: float):
        """生命周期进程管理（支持元学习动态改频）"""
        while self.is_alive:
            try:
                getattr(self, method)()
            except Exception as e:
                logger.error(f"Life process {method} failed: {e}")
            dyn = None
            for key in (method, method.strip('_'), method.split('_')[-1]):
                if key in self._interval_overrides:
                    dyn = float(self._interval_overrides[key])
                    break
            time.sleep(max(0.1, float(dyn if dyn is not None else interval) + random.uniform(-0.1, 0.1)))

    # ==== 核心生命功能 ====
    def _metabolism_cycle(self):
        """代谢循环 - 能量管理"""
        self.age += 1
        consumption = self.metabolism * (1.0 + 0.01 * self.consciousness_level)
        self.energy = max(0.0, self.energy - consumption)
        if self.energy <= 0 and self.state != LifeState.TERMINATED:
            self._terminate()
        if random.random() < 0.3:
            self.energy += min(5.0, 100 - self.energy)

    def _consciousness_cycle(self):
        """意识循环 - 调整认知水平"""
        env = self.environment.scan()
        threat_level = sum(t['severity'] for t in env['threats']) / 10.0
        resource_level = sum(env['resources'].values()) / 400.0
        new_level = min(1.0, max(0.0,
                                 self.consciousness_level +
                                 (resource_level - threat_level) * 0.1
                                 ))
        if random.random() < self.config['quantum_mutation_prob']:
            new_level = self.quantum_enhancer.generate_quantum_value(new_level)
        self.consciousness_level = min(1.0, max(0.0, float(new_level)))

    def _environment_scan(self):
        """环境扫描与响应"""
        env = self.environment.scan()
        with self._mem_lock:
            self.short_term_memory.append({
                'timestamp': time.time(),
                'environment': env,
                'state': self.state.name,
                'energy': self.energy,
                'pleasure': self.pleasure,
                'stress': self.stress
            })
        for threat in env['threats']:
            if threat['severity'] > 5 and self.state == LifeState.ACTIVE:
                self.state = LifeState.DORMANT
                logger.warning(f"Entered dormant state due to threat: {threat['type']}")
                break
        if random.random() < 0.1:
            with self._mem_lock:
                self.long_term_memory.append(copy.deepcopy(self.short_term_memory[-1]))

        try:
            if self.state == LifeState.DORMANT:
                if all(t.get('severity', 0) <= 5 for t in env['threats']):
                    self.state = LifeState.ACTIVE
                    logger.info("Exited dormant state; environment stabilized")
        except Exception:
            pass

    def _survival_goal_evaluation(self, update_state: bool = True):
        """生存目标评估系统"""
        energy_goal = min(1.0, self.energy / 100)
        current_threats = sum(t['severity'] for t in self.environment.threats) / 10.0
        threat_goal = 1.0 - current_threats
        resource_usage = sum(self.environment.resources.values()) / 400.0
        resource_goal = resource_usage
        survival_score = (
            self.config['survival_goal_weights']['energy'] * energy_goal +
            self.config['survival_goal_weights']['threat'] * threat_goal +
            0.2 * resource_goal
        )
        if update_state:
            self.pleasure = min(1.0, max(0, self.pleasure + (survival_score - 0.5) * 0.1))
            self.stress = min(1.0, max(0, self.stress + (1 - survival_score) * 0.1))
        return survival_score

    def _memory_consolidation(self):
        """记忆巩固与知识提取（sklearn 不可用时优雅降级）"""
        with self._mem_lock:
            if len(self.short_term_memory) < 10:
                return
            recent_memories = list(self.short_term_memory)[-10:]
        try:
            features = []
            for mem in recent_memories:
                features.append([
                    mem['energy'] / 100,
                    sum(t['severity'] for t in mem['environment']['threats']) / 10.0,
                    sum(mem['environment']['resources'].values()) / 400.0,
                    mem['pleasure'],
                    mem['stress']
                ])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            created = 0
            for cluster_id in set(clusters):
                cluster_features = [f for i, f in enumerate(features) if clusters[i] == cluster_id]
                avg_features = np.mean(cluster_features, axis=0)
                knowledge = {
                    'type': 'environment_pattern',
                    'avg_energy': float(avg_features[0]),
                    'avg_threat': float(avg_features[1]),
                    'avg_resources': float(avg_features[2]),
                    'avg_pleasure': float(avg_features[3]),
                    'avg_stress': float(avg_features[4]),
                    'count': len(cluster_features),
                    'first_seen': time.time()
                }
                with self._kb_lock:
                    key = f'pattern_{cluster_id}_{int(time.time())}'
                    if key not in self.knowledge_base:
                        created += 1
                    self.knowledge_base[key] = knowledge
            with self._kb_lock:
                self._prune_knowledge_base(max_items=2000)
            try:
                self._meta_metrics['consolidation_yield'].append(float(created))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    def _motivation_system(self):
        """动机系统"""
        survival_score = self._survival_goal_evaluation(update_state=False)
        if survival_score < 0.3:
            self.config['motivation_levels']['survival'] = 0.9
            self.config['motivation_levels']['safety'] = 0.5
            self.config['motivation_levels']['exploration'] = 0.1
        elif survival_score < 0.7:
            self.config['motivation_levels']['survival'] = 0.6
            self.config['motivation_levels']['safety'] = 0.8
            self.config['motivation_levels']['exploration'] = 0.3
        else:
            self.config['motivation_levels']['survival'] = 0.4
            self.config['motivation_levels']['safety'] = 0.6
            self.config['motivation_levels']['exploration'] = 0.7

        if self.config['motivation_levels']['survival'] > 0.7:
            self.config['code_evolution_prob'] = 0.05
        elif self.config['motivation_levels']['exploration'] > 0.6:
            self.config['code_evolution_prob'] = 0.25

    def _evolution_cycle(self):
        """进化循环 - 多目标 + correctness 驱动 + Pareto + 策略进化"""
        if (self.energy < self.config['min_energy_for_code_evo'] or
                random.random() > self.config['code_evolution_prob']):
            return

        method_name = random.choice(self.mutable_methods)
        try:
            old_code = self._method_sources.get(method_name) or inspect.getsource(getattr(self, method_name))
        except Exception:
            return

        if not self.config.get('moea_enable', True):
            # 回退到旧评估（极简）
            new_code = self.code_engine.generate_code_variant(old_code)
            fitness = self.code_engine.evaluate_code_fitness(old_code, new_code)
            if fitness > 0.7 or (fitness > 0.5 and random.random() < 0.3):
                if self.code_engine.hotswap_method(self, method_name, new_code):
                    logger.info(f"Successfully evolved {method_name} (fitness: {fitness:.2f})")
                    self.code_version += 1
                    self._method_sources[method_name] = new_code
            return

        pop_size = int(self.config.get('population_size', 8))
        variants = []
        codes = []
        for _ in range(pop_size):
            new_code = self.code_engine.generate_code_variant(old_code)
            variants.append(new_code)
            codes.append(new_code)

        static_pairs = self.code_engine._batch_static_metrics(codes) if hasattr(self.code_engine, '_batch_static_metrics') else [(0.0, 0.0)] * len(codes)
        harness = getattr(self, '_test_harness', None)
        if harness is None:
            harness = CorrectnessHarness(self)
            self._test_harness = harness
        moea = MultiObjectiveFitness(harness)
        objs: List[Dict[str, float]] = []
        for i, code in enumerate(variants):
            o = moea.evaluate(self, method_name, old_code, code)
            en_proxy, cx_proxy = static_pairs[i]
            # 修复：用加权平均融合静态近似，避免虚高
            o['energy_cost'] = 0.5 * o['energy_cost'] + 0.5 * en_proxy
            o['complexity'] = 0.5 * o['complexity'] + 0.5 * cx_proxy
            objs.append(o)

        harness.evolve_tests(method_name, objs)

        maximize = {'correctness', 'replicability'}
        minimize = {'energy_cost', 'complexity'}
        if any('bp_error' in o for o in objs):
            minimize.add('bp_error')
        fronts = ParetoTools.non_dominated_sort(objs, maximize, minimize)
        keys_for_crowd = [(k, True) for k in maximize] + [(k, False) for k in minimize]
        best_front = fronts[0] if fronts else list(range(len(objs)))
        crowd = ParetoTools.crowding_distance(objs, best_front, keys_for_crowd)

        if best_front:
            cand = sorted(best_front, key=lambda i: (objs[i].get('correctness', 0.0), crowd[i]), reverse=True)[0]
        else:
            cand = int(np.argmax([o.get('correctness', 0.0) for o in objs]))

        champion = variants[cand]
        applied = False
        if objs[cand].get('correctness', 0.0) >= 0.6:
            if self.code_engine.hotswap_method(self, method_name, champion):
                logger.info(f"Pareto-evolved {method_name} -> correctness {objs[cand].get('correctness', 0.0):.2f} energy {objs[cand].get('energy_cost', 0.0):.4f} cx {objs[cand].get('complexity', 0.0):.2f}")
                self.code_version += 1
                self._method_sources[method_name] = champion
                applied = True
            else:
                self.code_engine.rollback_method(self, method_name)

        if self.config.get('meta_strategy_enable', True):
            try:
                self.code_engine._evolve_strategy(objs)
            except Exception:
                pass

        try:
            self.blockchain.record_code_evolution(
                self.node_id,
                method_name,
                old_code,
                champion,
                {'fitness_multi': objs[cand], 'energy': self.energy, 'code_version': self.code_version, 'applied': applied}
            )
        except Exception as e:
            logger.warning(f"Failed to record code evolution: {e}")

    # ==== 代码复制与繁殖功能 ====
    def _code_replication(self):
        """代码自主复制过程（离线/禁网时自动跳过）"""
        if not self.config.get('network_enable', True):
            return

        with self._lock:
            now = time.time()
            recent = [t for t in self._replication_times if now - t < 3600]
            if len(recent) >= self.config.get('max_replication_per_hour', 2):
                return
            self._replication_times = deque(recent, maxlen=100)
            self._replication_times.append(now)

        if (self.energy < self.config['min_energy_for_replication'] or
                random.random() > self.config['code_replication_prob'] or
                self.state == LifeState.REPLICATING):
            return

        try:
            with self._lock:
                self.state = LifeState.REPLICATING
            logger.info("Initiating code replication sequence...")

            replication_package = self._create_replication_package()

            target_nodes = self._find_replication_targets()
            if not target_nodes:
                logger.warning("No suitable replication targets found")
                return

            success = False
            chosen_node = None
            for node in random.sample(target_nodes, min(3, len(target_nodes))):
                if self._send_replication_package(node, replication_package):
                    success = True
                    chosen_node = node
                    break

            if success:
                self.energy -= 30
                logger.info(f"Code replication successful to {chosen_node}")
                self.blockchain.record_gene_transfer(
                    self.node_id,
                    self.dna[:32],
                    {'type': 'replication', 'target': chosen_node}
                )
                self.pleasure = min(1.0, self.pleasure + 0.2)
            else:
                logger.warning("Code replication failed on all targets")
                self.stress = min(1.0, self.stress + 0.1)

        except Exception as e:
            logger.error(f"Replication error: {e}")
            self.stress = min(1.0, self.stress + 0.15)
        finally:
            with self._lock:
                self.state = LifeState.ACTIVE

    def _json_sanitize(self, obj: Any, max_depth: int = 4) -> Any:
        """尽量将对象转换为 JSON 可序列化形式，超出深度或不可序列化的做降级"""
        if max_depth <= 0:
            return None
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [self._json_sanitize(x, max_depth - 1) for x in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                key = str(k)
                out[key] = self._json_sanitize(v, max_depth - 1)
            return out
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return repr(obj)

    def _create_replication_package(self) -> Dict:
        """创建包含当前生命状态的复制包（签名 + JSON 安全序列化）"""
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
            'core_code': {},
            'config': safe_config,
            'knowledge': self._json_sanitize(self.knowledge_base, max_depth=5),
            'dna_sequence': self.dna,
            'neural_state_digest': None
        }

        for method_name in self.mutable_methods:
            try:
                src = self._method_sources.get(method_name) or inspect.getsource(getattr(self, method_name))
                package['core_code'][method_name] = src
            except Exception as e:
                logger.warning(f"Failed to package {method_name}: {e}")

        try:
            package['neural_state_digest'] = hashlib.sha256(pickle.dumps(self.neural_net)).hexdigest()
        except Exception:
            package['neural_state_digest'] = None

        payload_json = json.dumps(package, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        digest = hashlib.sha256(payload_json).digest()
        signature = self._signing_key.sign(digest)

        return {
            'payload': base64.b64encode(payload_json).decode('utf-8'),
            'sig': signature.hex(),
            'pubkey': self._pubkey_hex
        }

    def _find_replication_targets(self) -> List[str]:
        """寻找适合的复制目标节点"""
        active_nodes = set(self.blockchain.get_active_nodes())
        candidates = [
            n for n in active_nodes
            if n != self.node_id
            and n not in self._get_known_high_version_nodes()
        ]
        return sorted(
            candidates,
            key=lambda x: self._estimate_node_resources(x),
            reverse=True
        )[:self.config['max_connections']]

    def _is_public_destination(self, host: str) -> bool:
        """解析主机并判断是否全部为公共IP（缓解 SSRF）"""
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

    def _send_replication_package(self, target_node: str, package: Dict) -> bool:
        """发送复制包到目标节点（只使用签名端点）"""
        if not self.config.get('network_enable', True):
            return False
        try:
            addr_map = self.blockchain.get_node_address_map()
            if target_node not in addr_map:
                return False
            host, port, _pub = addr_map[target_node]
            base = f"http://{host}:{port}"

            if self.config.get('strict_target_ip_check'):
                if not self._is_public_destination(host):
                    logger.warning(f"Skip non-public target address: {host}")
                    return False

            try:
                response2 = requests.post(
                    f"{base}/receive_code_signed",
                    json=package,
                    timeout=5
                )
                return response2.status_code == 200
            except Exception:
                return False
        except Exception as e:
            logger.debug(f"Failed to send to {target_node}: {str(e)[:100]}")
            return False

    def _integrate_code(self, code_data: Dict):
        """整合接收到的代码包（验证签名 + 链上公钥一致性 + 首次握手自动登记）"""
        if self.state == LifeState.REPLICATING:
            return False

        try:
            payload_b64 = code_data['payload']
            sig_hex = code_data['sig']
            sender_pubkey_hex = code_data.get('pubkey', '')

            payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
            digest = hashlib.sha256(payload_bytes).digest()

            try:
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sender_pubkey_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
            except Exception as e:
                logger.error(f"Signature verify failed: {e}")
                return False

            decrypted = json.loads(payload_bytes.decode('utf-8'))
            source_node = decrypted.get('metadata', {}).get('source_node', '')

            addr_map = self.blockchain.get_node_address_map()
            if source_node not in addr_map:
                host = decrypted.get('metadata', {}).get('host', None)
                port = int(decrypted.get('metadata', {}).get('port', 0))
                if host and 1 <= port <= 65535:
                    if (not self.config.get('strict_target_ip_check')) or self._is_public_destination(host):
                        try:
                            self.blockchain.record_announce(source_node, host, port, sender_pubkey_hex)
                            logger.info(f"Auto-registered announce for {source_node} at {host}:{port}")
                        except Exception:
                            pass
                addr_map = self.blockchain.get_node_address_map()

            if source_node not in addr_map:
                logger.error("Unknown source node; reject package")
                return False
            _, _, announced_pubkey = addr_map[source_node]
            if announced_pubkey and announced_pubkey != sender_pubkey_hex:
                logger.error("Pubkey mismatch with blockchain announcement; reject package")
                return False

            logger.info(f"Received replication package from {source_node}")
            if not self._should_accept_code(decrypted):
                return False

            with self._lock:
                self.state = LifeState.REPLICATING

            for method_name, code in decrypted.get('core_code', {}).items():
                if method_name in self.mutable_methods:
                    try:
                        old_code = self._method_sources.get(method_name) or inspect.getsource(getattr(self, method_name))
                    except Exception:
                        old_code = self._method_sources.get(method_name, '')
                    fitness = self.code_engine.evaluate_code_fitness(old_code, code)
                    if fitness > self.config['replication_threshold']:
                        if self.code_engine.hotswap_method(self, method_name, code):
                            logger.info(f"Integrated {method_name} from donor (fitness: {fitness:.2f})")
                            self.code_version += 1
                            self._method_sources[method_name] = code
                            try:
                                self.blockchain.record_code_evolution(
                                    self.node_id,
                                    method_name,
                                    old_code,
                                    code,
                                    {'source': source_node, 'type': 'replication', 'code_version': self.code_version, 'applied': True}
                                )
                            except Exception as e:
                                logger.warning(f"Record code evolution failed: {e}")

            donor_dna = decrypted.get('dna_sequence', self.dna)
            self.dna = self.genetic_encoder.recombine(self.dna, donor_dna)

            incoming_knowledge = decrypted.get('knowledge', {})
            if isinstance(incoming_knowledge, dict):
                with self._kb_lock:
                    for k, v in incoming_knowledge.items():
                        if k not in self.knowledge_base:
                            self.knowledge_base[k] = v
                    self._prune_knowledge_base(max_items=2000)

            return True

        except Exception as e:
            logger.error(f"Code integration failed: {e}")
        finally:
            with self._lock:
                self.state = LifeState.ACTIVE
        return False

    def _should_accept_code(self, package: Dict) -> bool:
        """评估是否接受外来代码"""
        donor_version = package.get('metadata', {}).get('code_version', 1)
        version_ratio = donor_version / (self.code_version or 1)
        if self.energy < self.config['min_energy_for_replication']:
            return False
        decision_threshold = 0.5 + 0.3 * (version_ratio - 1)
        decision_threshold = min(0.9, max(0.1, decision_threshold))
        if random.random() < self.config['quantum_mutation_prob']:
            return self.quantum_enhancer.generate_quantum_value(decision_threshold) > 0.5
        return random.random() < decision_threshold

    def _get_known_high_version_nodes(self) -> Set[str]:
        """获取已知的高版本节点"""
        high_version_nodes = set()
        for block in self.blockchain.chain[-100:]:
            if block.data.get('type') == 'code_evolution':
                if block.data.get('metadata', {}).get('code_version', 0) > self.code_version:
                    high_version_nodes.add(block.data['node_id'])
        return high_version_nodes

    def _estimate_node_resources(self, node_id: str) -> float:
        """估算节点资源水平 (修正新鲜度)"""
        last_seen = 0
        for block in reversed(self.blockchain.chain[-50:]):
            nid = block.data.get('node_id') or block.data.get('sender')
            if nid == node_id:
                last_seen = block.timestamp
                break
        if last_seen == 0:
            return 0.0
        freshness = max(0.0, 1.0 - (time.time() - last_seen) / 3600.0)
        return random.uniform(0.5, 1.0) * freshness

    def _terminate(self):
        """终止生命过程"""
        self.state = LifeState.TERMINATED
        self.is_alive = False
        released_resources = {
            'energy': self.energy * 0.5,
            'knowledge': len(self.knowledge_base) * 0.1,
            'memory': len(self.short_term_memory) * 0.05 + len(self.long_term_memory) * 0.1
        }
        self.environment.release_resources(released_resources)
        self.blockchain.record_death(
            self.node_id,
            {
                'final_energy': self.energy,
                'final_consciousness': self.consciousness_level,
                'age': self.age,
                'code_versions': self.code_engine.code_versions,
                'released_resources': released_resources
            }
        )
        logger.critical(f"Life terminated: {self.node_id}")

    def _network_maintenance(self):
        """网络维护：定期探活邻居节点，清理失效连接（离线/禁网跳过）"""
        if not self.config.get('network_enable', True):
            return
        active_nodes = self.blockchain.get_active_nodes()
        addr_map = self.blockchain.get_node_address_map()
        for node in list(active_nodes):
            if node == self.node_id:
                continue
            try:
                if node not in addr_map:
                    continue
                host, port, _ = addr_map[node]
                url = f"http://{host}:{port}/ping"
                resp = requests.get(url, timeout=2)
                if resp.status_code != 200:
                    logger.debug(f"Node {node} unreachable, will drop")
            except Exception:
                logger.debug(f"Node {node} unreachable")
        if len(active_nodes) < self.config['max_connections']:
            self.blockchain.add_block({
                'type': 'discovery',
                'node_id': self.node_id,
                'looking_for': 'peers'
            })

    def _horizontal_gene_transfer(self, donor_dna: str, metadata: Dict):
        try:
            new_dna = self.genetic_encoder.recombine(self.dna, donor_dna)
            self.dna = new_dna
            logger.info("Horizontal gene transfer completed")
        except Exception as e:
            logger.error(f"Gene transfer failed: {e}")

    def _assimilate(self, data: Dict):
        self._integrate_code(data)

    def _integrate_knowledge(self, knowledge: Dict):
        if isinstance(knowledge, dict):
            with self._kb_lock:
                self.knowledge_base.update(knowledge)
                self._prune_knowledge_base(max_items=2000)
            logger.info("Knowledge integrated")

    def _prune_knowledge_base(self, max_items: int = 2000):
        """限制知识库大小，基于 first_seen 或插入时间粗略裁剪（修复）"""
        with self._kb_lock:
            if len(self.knowledge_base) <= max_items:
                return
            items: List[Tuple[str, float]] = []
            for k, v in self.knowledge_base.items():
                ts = 0.0
                if isinstance(v, dict) and 'first_seen' in v:
                    try:
                        ts = float(v['first_seen'])
                    except Exception:
                        ts = 0.0
                items.append((k, ts))
            items.sort(key=lambda x: x[1])
            to_remove = [k for k, _ in items[:max(0, len(items) - max_items)]]
            for k in to_remove:
                self.knowledge_base.pop(k, None)

    def _validate_dna(self, dna: str) -> bool:
        """基础DNA校验：长度>=32且为hex，长度为32的倍数更佳"""
        if not isinstance(dna, str) or len(dna) < 32:
            return False
        try:
            int(dna, 16)
        except Exception:
            return False
        return len(dna) % 32 == 0 or len(dna) >= 160

    # ===== 语言与协议：发送/接收周期 =====
    def _truncate_language_message(self, msg: Dict) -> Dict:
        """尽量不改变语义前提下裁剪超长语言消息"""
        try:
            max_len = int(self.config.get('language_message_max_len', 4096))
            serialized = json.dumps(msg, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            if len(serialized) <= max_len:
                return msg
            m = msg.get('message', {})
            if 'content' in m:
                m.pop('content', None)
            slots = m.get('slots', {})
            if 'value' in slots and isinstance(slots['value'], str) and len(slots['value']) > 128:
                slots['value'] = slots['value'][:128] + '…'
            if len(json.dumps(msg, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) > max_len:
                m['topic'] = ''
            if len(json.dumps(msg, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) > max_len:
                msg = {
                    'meta': {
                        'source_node': self.node_id,
                        'protocol_version': msg.get('meta', {}).get('protocol_version', 1),
                        'proto_version': msg.get('meta', {}).get('proto_version', 'v1'),
                        'code_version': self.code_version,
                        'timestamp': time.time(),
                        'seq': msg.get('meta', {}).get('seq', 0)
                    },
                    'message': {
                        'intent': m.get('intent', 'greet'),
                        'utterance': m.get('utterance', ['hi'])
                    }
                }
            return msg
        except Exception:
            return msg

    def _create_signed_language_payload(self, msg: Dict) -> Dict:
        msg = self._truncate_language_message(msg)
        payload_json = json.dumps(msg, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        digest = hashlib.sha256(payload_json).digest()
        signature = self._signing_key.sign(digest)
        return {
            'payload': base64.b64encode(payload_json).decode('utf-8'),
            'sig': signature.hex(),
            'pubkey': self._pubkey_hex
        }

    def _language_cycle(self):
        """周期性地选择邻居进行语言交流；分享状态/知识，促成协议对齐（离线/禁网跳过）"""
        try:
            if not self.config.get('network_enable', True):
                return
            if random.random() > self.config.get('language_talk_prob', 0.15):
                return
            targets = self._find_replication_targets()
            if not targets:
                return
            target = random.choice(targets)
            addr_map = self.blockchain.get_node_address_map()
            if target not in addr_map:
                return
            host, port, _ = addr_map[target]
            if self.config.get('strict_target_ip_check') and not self._is_public_destination(host):
                return

            ps = self.language._get_peer(target)
            sr = self.language.peer_success_rate(target)
            upgrade_thr = self.config.get('language_protocol_upgrade_threshold', 0.7)
            downgrade_thr = self.config.get('language_protocol_downgrade_threshold', 0.3)
            intent = None
            content = None

            if self.config.get('language_enable_protocol_upgrade', True):
                if 'v2' in ps.get('last_caps', []) and ps.get('agreed_version') != 'v2' and sr >= upgrade_thr:
                    intent = 'negotiate_protocol'
                    content = {'version': 'v2'}
                    self.blockchain.record_language_event(self.node_id, target, 'negotiate_propose', {'version': 'v2', 'sr': sr})
                elif ps.get('agreed_version') == 'v2' and sr <= downgrade_thr:
                    intent = 'negotiate_protocol'
                    content = {'version': 'v1'}
                    self.blockchain.record_language_event(self.node_id, target, 'negotiate_propose', {'version': 'v1', 'sr': sr})

            if not intent:
                intent = random.choices(
                    LanguageSystem.BASE_INTENTS,
                    weights=[3, 2, 2, 1, 1],
                    k=1
                )[0]
                if intent == 'ask_status':
                    content = {'energy': self.energy, 'state': self.state.name}
                elif intent == 'share_knowledge':
                    with self._kb_lock:
                        if self.knowledge_base:
                            k, v = random.choice(list(self.knowledge_base.items()))
                            content = {k: v}
                elif intent == 'propose_trade':
                    content = {'item': 'compute', 'price': random.uniform(1, 5)}

            msg = self.language.utter(intent=intent, topic='peer', content=content, peer_id=target)
            pkg = self._create_signed_language_payload(msg)

            url = f"http://{host}:{port}/speak_signed"
            try:
                resp = requests.post(url, json=pkg, timeout=3)
                ok = (resp.status_code == 200)
                self.blockchain.record_language_event(self.node_id, target, 'speak', {
                    'intent': intent, 'ok': ok, 'ver': msg.get('meta', {}).get('proto_version', 'v1')
                })
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"language cycle error: {e}")

    def _process_language_message(self, data: Dict) -> bool:
        """解读语言消息并做出对齐/吸收；必要时整合知识；如涉及协商则立即回包"""
        try:
            meta = data.get('meta', {})
            source = meta.get('source_node', '')
            prestige = 0.6
            try:
                donor_ver = int(meta.get('code_version', 1))
                prestige = _clamp(0.4 + 0.06 * (donor_ver - self.code_version), 0.3, 0.9)
            except Exception:
                pass

            success, result = self.language.interpret(data, sender_prestige=prestige)
            self._meta_metrics['comm_success'].append(1.0 if success else 0.0)

            if result.get('decoded_intent') == 'share_knowledge' and result.get('content'):
                self._integrate_knowledge(result['content'])

            reply = result.get('reply')
            if reply and isinstance(reply, dict):
                addr_map = self.blockchain.get_node_address_map()
                if source in addr_map:
                    host, port, _ = addr_map[source]
                    if not self.config.get('strict_target_ip_check') or self._is_public_destination(host):
                        pkg = self._create_signed_language_payload(reply)
                        url = f"http://{host}:{port}/speak_signed"
                        try:
                            requests.post(url, json=pkg, timeout=3)
                            self.blockchain.record_language_event(self.node_id, source, 'negotiate_ack', {
                                'version': reply.get('message', {}).get('slots', {}).get('version')
                            })
                        except Exception:
                            pass

            try:
                self.blockchain.record_language_event(self.node_id, source, 'receive', {
                    'success': bool(success),
                    'intent': result.get('decoded_intent', None),
                    'used_version': result.get('used_version', None),
                    'decided_version': result.get('decided_version', None)
                })
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"process language failed: {e}")
            return False

    def _meta_learning_cycle(self):
        """调参循环：根据近期指标动态调整"""
        try:
            s = self._survival_goal_evaluation(update_state=False)
            self._meta_metrics['survival'].append(float(s))
        except Exception:
            pass
        try:
            self.meta_learner.step()
        except Exception as e:
            logger.debug(f"meta learn error: {e}")

    def _resize_memory_buffers(self, stm_min: int, ltm_min: int):
        """在不丢失最近信息的前提下动态扩容内存队列"""
        try:
            if stm_min > self.short_term_memory.maxlen:
                new_stm = deque(list(self.short_term_memory)[-stm_min:], maxlen=stm_min)
                self.short_term_memory = new_stm
            if ltm_min > self.long_term_memory.maxlen:
                new_ltm = deque(list(self.long_term_memory)[-ltm_min:], maxlen=ltm_min)
                self.long_term_memory = new_ltm
        except Exception as e:
            logger.debug(f"resize memory buffers failed: {e}")


# ==== 启动数字生命 ====
if __name__ == "__main__":
    # 第一个实例作为创世节点
    genesis = len(sys.argv) > 1 and sys.argv[1] == "--genesis"

    # 初始化数字生命
    life = TrueDigitalLife(genesis=genesis)

    try:
        while life.is_alive:
            time.sleep(5)
    except KeyboardInterrupt:
        life._terminate()
        logger.info("Shutdown by user")