import textwrap
import ast
import copy
import difflib
import hashlib
import importlib
import inspect
import json
import logging
import os
import pickle
import psutil
import random
import requests
import socket
import sys
import tempfile
import threading
import time
import types
import uuid
import linecache
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Deque, Callable

import numpy as np
from cryptography.fernet import Fernet
from flask import Flask, jsonify, request
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 新增依赖
import secrets
from functools import wraps
import base64
try:
    import astor
except Exception:
    astor = None

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

# 可选工具依赖（优雅降级）
try:
    from mypy import api as mypy_api
except Exception:
    mypy_api = None

try:
    from radon.metrics import mi_visit
except Exception:
    mi_visit = None

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('digital_life.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TrueDigitalLife')


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


# 动态适应度评估系统（基础）
class DynamicFitnessEvaluator:
    """动态调整的适应度评估系统"""
    def __init__(self):
        self.metrics = {
            'functionality': 0.5,
            'novelty': 0.3,
            'complexity': 0.2,
            'energy_efficiency': 0.4,
            'replicability': 0.3
        }
        self.adaptive_weights = {
            'stable': {'functionality': 0.6, 'energy_efficiency': 0.4},
            'explore': {'novelty': 0.7, 'complexity': 0.3},
            'replicate': {'replicability': 0.8, 'functionality': 0.2}
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
            'replicability': self._replicability_score(mutated)
        }
        return float(sum(scores[k] * weights.get(k, 0) for k in scores) / max(1e-9, sum(weights.values())))

    def _select_evaluation_mode(self, context) -> str:
        if context.get('energy', 100) < 30:
            return 'stable'
        if context.get('stagnation', 0) > 5:
            return 'explore'
        if context.get('replication_mode', False):
            return 'replicate'
        return 'stable'

    def _functionality_score(self, original: str, mutated: str) -> float:
        try:
            compile(mutated, '<string>', 'exec')
            return 0.9 + 0.1 * random.random()
        except Exception:
            return 0.1

    def _novelty_score(self, original: str, mutated: str) -> float:
        if original == mutated:
            return 0.0
        matcher = difflib.SequenceMatcher(None, original, mutated)
        return 1 - matcher.ratio()

    def _complexity_score(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            complexity = len(list(ast.walk(tree))) / 100
            return min(1.0, complexity)
        except Exception:
            return 0.5

    def _energy_efficiency_score(self, code: str) -> float:
        lines = code.splitlines()
        return 1.0 / (1 + len(lines) / 10)

    def _replicability_score(self, code: str) -> float:
        try:
            tree = ast.parse(code)
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


class LifeState(Enum):
    ACTIVE = auto()
    DORMANT = auto()
    REPLICATING = auto()
    EVOLVING = auto()
    TERMINATED = auto()


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

    def mine_block(self, difficulty: int):
        """工作量证明挖矿"""
        target = '0' * max(0, difficulty)
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()


# 安全沙箱：AST 安全检查与受限执行（可通过配置关闭）
class ASTSafetyError(Exception):
    pass


class ASTSafetyChecker(ast.NodeVisitor):
    FORBIDDEN_NODES = (
        ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Raise, ast.Delete, ast.Global, ast.Nonlocal
    )
    FORBIDDEN_NAMES = {
        '__import__', 'eval', 'exec', 'open', 'compile', 'input',
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
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
        'reversed': reversed, 'print': print,
    }

    @staticmethod
    def compile_and_load(src: str, func_name: str, unsafe: bool = False):
        filename = '<mutation-unsafe>' if unsafe else '<mutation>'

        # 让 inspect 能从内存中找到源码（对热更新方法可getsource）
        lines = [l if l.endswith('\n') else l + '\n' for l in src.splitlines()]
        linecache.cache[filename] = (len(src), None, lines, filename)

        if unsafe:
            code = compile(src, filename, 'exec')
            g = {}
            l = {}
            exec(code, g, l)
            fn = l.get(func_name) or g.get(func_name)
            if not isinstance(fn, types.FunctionType):
                raise ASTSafetyError(f'Function {func_name} not defined after exec')
            setattr(fn, '__source__', src)
            return fn

        tree = ast.parse(src)
        ASTSafetyChecker().visit(tree)
        code = compile(tree, filename, 'exec')
        g = {"__builtins__": SafeExec.ALLOWED_BUILTINS}
        l = {}
        exec(code, g, l)
        fn = l.get(func_name) or g.get(func_name)
        if not isinstance(fn, types.FunctionType):
            raise ASTSafetyError(f'Function {func_name} not defined after exec')
        setattr(fn, '__source__', src)
        return fn


# 神经网络AST转换器（未默认接入）
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


# ============ 新增：高级适应度/类型/LLM/元进化 ============
class LLMAdapter:
    """可选：语义级变异的外部接口。默认关闭，返回 None 表示不用。"""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled or bool(os.environ.get("EVOLVER_LLM_ENABLED"))

    def semantic_mutate(self, func_code: str, hint: str = "") -> Optional[str]:
        if not self.enabled:
            return None
        try:
            # TODO: 接入你的 LLM 服务，返回包含同名函数定义的源码
            return None
        except Exception:
            return None


class TypeChecker:
    """类型检查器：mypy 可选，无法使用时返回中性分"""
    def score(self, code: str) -> float:
        if mypy_api is None:
            return 0.6
        try:
            stdout, stderr, status = mypy_api.run(['-c', code, '--ignore-missing-imports'])
            return 1.0 if status == 0 else 0.3
        except Exception:
            return 0.6


class MultiObjectiveFitness:
    """多目标适应度聚合器：综合可读性/类型/原始各项"""
    def __init__(self):
        self.type_checker = TypeChecker()

    def readability_score(self, code: str) -> float:
        if mi_visit is None:
            return 0.6
        try:
            mi = mi_visit(code, True)
            return max(0.0, min(1.0, mi / 100.0))
        except Exception:
            return 0.6

    def aggregate(self, base_score: float, original: str, mutated: str) -> float:
        read = self.readability_score(mutated)
        type_s = self.type_checker.score(mutated)
        return float(0.55 * base_score + 0.20 * read + 0.25 * type_s)


class MetaEvolver:
    """小型遗传算法：进化变异操作符权重"""
    def __init__(self, operators: List[str], pop_size: int = 6, epoch: int = 20, noise: float = 0.2):
        self.ops = operators
        self.pop_size = pop_size
        self.epoch = epoch
        self.noise = noise
        self.population: List[Dict[str, float]] = [self._random_weights() for _ in range(pop_size)]
        self.scores = [0.0 for _ in range(pop_size)]
        self.usage = [0 for _ in range(pop_size)]
        self.idx = 0
        self.steps = 0

    def _random_weights(self) -> Dict[str, float]:
        w = {op: random.uniform(0.5, 1.5) for op in self.ops}
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    def current_weights(self) -> Dict[str, float]:
        return self.population[self.idx]

    def record(self, applied_ops: List[str], fitness_delta: float):
        self.scores[self.idx] += fitness_delta
        self.usage[self.idx] += 1
        self.steps += 1
        if self.steps % self.epoch == 0:
            self._evolve()
        self.idx = (self.idx + 1) % self.pop_size

    def _evolve(self):
        perf = [(i, (self.scores[i] / max(1, self.usage[i]))) for i in range(self.pop_size)]
        perf.sort(key=lambda x: x[1], reverse=True)
        top = [self.population[i] for i, _ in perf[: self.pop_size // 2]]

        new_pop = top.copy()
        while len(new_pop) < self.pop_size:
            p1, p2 = random.sample(top, 2)
            child = {}
            for op in self.ops:
                base = (p1[op] + p2[op]) / 2.0
                child[op] = max(1e-3, base * random.uniform(1 - self.noise, 1 + self.noise))
            s = sum(child.values())
            child = {k: v / s for k, v in child.items()}
            new_pop.append(child)

        self.population = new_pop
        self.scores = [0.0 for _ in range(self.pop_size)]
        self.usage = [0 for _ in range(self.pop_size)]


# ==== 根任务管理器：生存 + 种族延续 ====
class RootTaskManager:
    """
    Root 任务（存活 + 种族延续）评分与策略调节：
    - survival_ma：生存分移动平均（能量、威胁、稳定）
    - repro_rate：近一小时成功复制次数占目标上限的比例
    - unique_descendants：唯一后代节点数量（观测）
    - species_population：本链上最近 announce 的同物种节点数
    - root_score = 0.6 * survival_ma + 0.4 * repro_rate
    """
    def __init__(self, life: 'TrueDigitalLife'):
        self.life = life
        self.start_ts = time.time()
        self.survival_hist: Deque[float] = deque(maxlen=100)
        self.repro_events: Deque[float] = deque(maxlen=1000)
        self.unique_descendants: Set[str] = set()
        self.last_ledger_ts = 0.0
        self.root_score = 0.0
        self.survival_ma = 0.0
        self.repro_rate = 0.0
        self.species_population = 1

    def note_reproduction_success(self, target_node: str):
        now = time.time()
        self.repro_events.append(now)
        if target_node:
            self.unique_descendants.add(target_node)

    def _recent_repro_count(self, seconds: int = 3600) -> int:
        now = time.time()
        while self.repro_events and now - self.repro_events[0] > seconds:
            self.repro_events.popleft()
        return len(self.repro_events)

    def _compute_species_population(self) -> int:
        # 统计本地链上最近 announce 的同 species 节点数量
        species_id = getattr(self.life, 'species_id', '')
        seen: Dict[str, str] = {}
        for b in self.life.blockchain.chain:
            d = b.data
            if d.get('type') == 'announce':
                nid = d.get('node_id')
                sp = d.get('species', '')
                if nid:
                    seen[nid] = sp
        cnt = sum(1 for sp in seen.values() if sp == species_id)
        return max(1, cnt)

    def update(self):
        # 生存分：复用 survival_goal + 能量/威胁修正
        try:
            survival_score = self.life._survival_goal_evaluation()
        except Exception:
            # 回退：能量/威胁粗略估
            try:
                env = self.life.environment.scan()
                energy_norm = min(1.0, max(0.0, self.life.energy / 100.0))
                threat = sum(t['severity'] for t in env.get('threats', [])) / 10.0
                survival_score = max(0.0, min(1.0, 0.6 * energy_norm + 0.4 * (1.0 - threat)))
            except Exception:
                survival_score = 0.5

        self.survival_hist.append(float(survival_score))
        self.survival_ma = float(np.mean(self.survival_hist)) if self.survival_hist else survival_score

        # 繁殖速率
        recent = self._recent_repro_count(3600)
        target = max(1, self.life.config.get('max_replication_per_hour', 2))
        self.repro_rate = min(1.0, recent / target)

        # 物种规模
        self.species_population = self._compute_species_population()

        # 根任务总分
        self.root_score = float(0.6 * self.survival_ma + 0.4 * self.repro_rate)

    def adjust_policy(self):
        """根据根任务表现调参"""
        l = self.life
        # 生存差 -> 降低大改概率，保守
        if self.survival_ma < 0.45:
            l.config['code_evolution_prob'] = max(0.05, l.config.get('code_evolution_prob', 0.15) - 0.03)
            l.config['code_replication_prob'] = max(0.05, l.config.get('code_replication_prob', 0.1) - 0.02)
        # 繁殖低但能量高 -> 提升繁殖意愿，放宽整合阈值
        if self.repro_rate < 0.5 and l.energy > 60:
            l.config['code_replication_prob'] = min(0.5, l.config.get('code_replication_prob', 0.1) + 0.05)
            l.config['replication_threshold'] = max(0.55, l.config.get('replication_threshold', 0.7) - 0.02)
        # 种群较大且生存好 -> 适度探索
        if self.species_population >= 3 and self.survival_ma > 0.65:
            l.config['code_evolution_prob'] = min(0.35, l.config.get('code_evolution_prob', 0.15) + 0.03)

        # 周期性写链审计
        now = time.time()
        if now - self.last_ledger_ts > 60:
            try:
                l.blockchain.record_root_status(
                    l.node_id,
                    getattr(l, 'species_id', ''),
                    {
                        'root_score': self.root_score,
                        'survival_ma': self.survival_ma,
                        'repro_rate': self.repro_rate,
                        'unique_descendants': len(self.unique_descendants),
                        'species_population': self.species_population,
                        'energy': l.energy,
                        'age': l.age
                    }
                )
            except Exception:
                pass
            self.last_ledger_ts = now

    def kpis(self) -> Dict[str, Any]:
        return {
            'root_score': self.root_score,
            'survival_ma': self.survival_ma,
            'repro_rate': self.repro_rate,
            'unique_descendants': len(self.unique_descendants),
            'species_population': self.species_population,
            'uptime_sec': int(time.time() - self.start_ts)
        }


# 代码进化引擎
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

        # 变异操作符
        self.mutation_operators = {
            'control_flow': self._mutate_control_flow,
            'data_flow': self._mutate_data_flow,
            'api_call': self._mutate_api_calls,
            'quantum': self._quantum_mutation,
            'neural': self._neural_mutation,
            'replication': self._replication_mutation,
            'class_struct': self._mutate_class_struct,
            'semantic': self._semantic_mutation,
        }
        self.operator_weights = {op: 1.0 for op in self.mutation_operators}
        self.adaptive_mutation_rate = 0.2

        # 多目标适应度、LLM适配器、元进化器
        self.multiobj = MultiObjectiveFitness()
        self.llm = LLMAdapter()
        self.meta = MetaEvolver(list(self.mutation_operators.keys()))
        self._last_applied_ops: List[str] = []
        self._method_best_fitness: Dict[str, float] = {}
        self.method_fitness_history: Dict[str, Deque[float]] = {}

    def _get_method_source(self, method):
        try:
            return inspect.getsource(method)
        except Exception:
            return getattr(method, '__source__', None)

    def _dynamic_operator_selection(self) -> str:
        total = sum(self.operator_weights.values())
        r = random.uniform(0, total)
        upto = 0
        for op, weight in self.operator_weights.items():
            if upto + weight >= r:
                return op
            upto += weight
        return random.choice(list(self.mutation_operators.keys()))

    def _quantum_mutation(self, node: ast.AST) -> ast.AST:
        if random.random() < 0.05:
            if isinstance(node, ast.Constant):
                return ast.Constant(value=self.quantum.generate_quantum_value(node.value))
        return node

    def _neural_mutation(self, node: ast.AST) -> ast.AST:
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
                    left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_quantum_var', ctx=ast.Load()),
                    ops=[ast.NotEq()],
                    comparators=[ast.Constant(value=random.randint(0, 1))]
                )
            ]
        )
        node.test = augmented_test
        return node

    def _replication_mutation(self, node: ast.AST) -> ast.AST:
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
        if isinstance(node, ast.If):
            if random.random() < self.adaptive_mutation_rate:
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            if random.random() < self.adaptive_mutation_rate / 2:
                new_node = ast.If(
                    test=ast.Compare(
                        left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_quantum_var', ctx=ast.Load()),
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
                        test=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_break_flag', ctx=ast.Load()),
                        body=[ast.Break()],
                        orelse=[]
                    )
                )
        return node

    def _mutate_data_flow(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Assign):
            if random.random() < 0.2:
                node.value = ast.BinOp(
                    left=node.value,
                    op=ast.Add(),
                    right=ast.Constant(value=random.randint(0, 10))
                )
        return node

    def _mutate_api_calls(self, node: ast.AST) -> ast.AST:
        return node

    def _mutate_class_struct(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.ClassDef) and node.name == 'TrueDigitalLife':
            if random.random() < 0.4:
                new_fn = ast.FunctionDef(
                    name=f"_aux_homeostasis_{random.randint(0, 999)}",
                    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=[
                        ast.Assign(
                            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='pleasure', ctx=ast.Store())],
                            value=ast.Call(func=ast.Name(id='min', ctx=ast.Load()),
                                           args=[ast.Constant(value=1.0),
                                                 ast.BinOp(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='pleasure', ctx=ast.Load()),
                                                           op=ast.Add(),
                                                           right=ast.Constant(value=0.01))],
                                           keywords=[])),
                        ast.Return(value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='pleasure', ctx=ast.Load()))
                    ],
                    decorator_list=[]
                )
                node.body.append(new_fn)
        return node

    def _semantic_mutation(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.FunctionDef) and random.random() < 0.25:
            try:
                src = ast.unparse(node) if hasattr(ast, 'unparse') else None
                if not src:
                    return node
                hint = f"保持函数签名与行为，尽量更高效、可读性更好；函数名：{node.name}"
                new_src = self.llm.semantic_mutate(src, hint=hint)
                if new_src:
                    mod = ast.parse(textwrap.dedent(new_src))
                    for n in mod.body:
                        if isinstance(n, ast.FunctionDef) and n.name == node.name:
                            return n
            except Exception:
                pass
        return node

    def _parse_dna_to_mutations(self) -> List[Dict]:
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
                'energy_cost': (sum(ord(c) for c in segment) % 10) + 5
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

    class _OperatorApplier(ast.NodeTransformer):
        """将 engine 的操作符应用到 AST"""
        def __init__(self, engine: 'CodeEvolutionEngine'):
            self.engine = engine

        def visit(self, node):
            node = super().visit(node)
            # 使用元进化器提供的权重加权选择
            weights = self.engine.meta.current_weights()
            ops = list(self.engine.mutation_operators.keys())
            probs = [max(1e-9, weights[o]) for o in ops]
            s = sum(probs)
            probs = [p / s for p in probs]
            op = random.choices(ops, probs)[0]
            new_node = self.engine.mutation_operators[op](node)
            if new_node is not node:
                self.engine._last_applied_ops.append(op)
            return new_node

    def generate_code_variant(self, original_code: str) -> str:
        try:
            tree = ast.parse(original_code)
            tree = self._OperatorApplier(self).visit(tree)
            ast.fix_missing_locations(tree)
            if hasattr(ast, 'unparse'):
                new_code = ast.unparse(tree)
            elif astor is not None:
                new_code = astor.to_source(tree)
            else:
                raise RuntimeError("Cannot unparse AST: ast.unparse/astor not available")
            new_code = textwrap.dedent(new_code)
            if random.random() < 0.1:
                new_code = self._post_process_mutation(new_code)
            return new_code
        except Exception as e:
            logger.error(f"Enhanced mutation failed: {e}")
            return original_code

    def _post_process_mutation(self, code: str) -> str:
        lines = code.splitlines()
        if len(lines) > 3 and random.random() < 0.3:
            insert_pos = random.randint(0, len(lines) - 1)
            new_line = f"# Mutated by quantum effect at {time.time()}"
            lines.insert(insert_pos, new_line)
        return '\n'.join(lines)

    def evaluate_code_fitness(self, original: str, mutated: str, context: dict, method_name: Optional[str] = None) -> float:
        base = self.fitness_evaluator.evaluate(original, mutated, context)
        final = self.multiobj.aggregate(base, original, mutated)
        if method_name:
            prev_best = self._method_best_fitness.get(method_name, 0.0)
            delta = final - prev_best
            self._method_best_fitness[method_name] = max(prev_best, final)
            if self._last_applied_ops:
                self.meta.record(self._last_applied_ops, delta)
                self._last_applied_ops.clear()
            hist = self.method_fitness_history.setdefault(method_name, deque(maxlen=50))
            hist.append(final)
        return float(final)

    def _calculate_complexity(self, code: str) -> float:
        try:
            tree = ast.parse(code)
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
        lines1 = code1.splitlines()
        lines2 = code2.splitlines()
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        return 1 - matcher.ratio()

    def hotswap_method(self, instance, method_name: str, new_code: str, unsafe: bool = False) -> bool:
        try:
            new_method = SafeExec.compile_and_load(new_code, method_name, unsafe=unsafe)
            if method_name not in self.backup_methods:
                self.backup_methods[method_name] = getattr(instance, method_name)
            setattr(instance, method_name, types.MethodType(new_method, instance))
            self.code_versions[method_name] = self.code_versions.get(method_name, 0) + 1
            # 确保可变方法映射指向最新方法
            if hasattr(instance, 'mutable_methods') and method_name in instance.mutable_methods:
                instance.mutable_methods[method_name] = getattr(instance, method_name)
            return True
        except Exception as e:
            logger.error(f"Method hotswap failed: {e}")
            return False

    def rollback_method(self, instance, method_name: str) -> bool:
        if method_name in self.backup_methods:
            setattr(instance, method_name, self.backup_methods[method_name])
            logger.info(f"Rolled back {method_name}")
            return True
        return False


class DistributedLedger:
    """为数字生命定制的区块链系统"""
    def __init__(self, node_id: str, genesis: bool = False, difficulty: int = 2):
        self.chain: List[Block] = []
        self.node_id = node_id
        self.difficulty = difficulty
        self.pending_transactions: List[Dict] = []
        self._lock = threading.RLock()

        os.makedirs('chaindata', exist_ok=True)

        if genesis or not self.load_chain():
            self.create_genesis_block()

    def create_genesis_block(self):
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
        tmp_path = f'chaindata/{self.node_id}_chain.pkl.tmp'
        final_path = f'chaindata/{self.node_id}_chain.pkl'
        with open(tmp_path, 'wb') as f:
            pickle.dump(chain_data, f)
        os.replace(tmp_path, final_path)

    def load_chain(self) -> bool:
        try:
            with self._lock:
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
        data = {
            'type': 'gene_transfer',
            'sender': sender,
            'dna_fragment': dna_fragment[:32],
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.add_block(data)

    def record_evolution(self, node_id: str, old_dna: str, new_dna: str, metadata: Dict):
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
        data = {
            'type': 'death',
            'node_id': node_id,
            'final_state': final_state,
            'timestamp': time.time()
        }
        self.add_block(data)

    def record_announce(self, node_id: str, host: str, port: int, pubkey_hex: str, species: str = ''):
        data = {
            'type': 'announce',
            'node_id': node_id,
            'host': host,
            'port': port,
            'pubkey': pubkey_hex,
            'species': species,
            'timestamp': time.time()
        }
        self.add_block(data)

    def record_root_status(self, node_id: str, species_id: str, metrics: Dict[str, Any]):
        data = {
            'type': 'root_status',
            'node_id': node_id,
            'species': species_id,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.add_block(data)

    def get_active_nodes(self) -> List[str]:
        active_nodes = set()
        with self._lock:
            for block in self.chain:
                data = block.data
                if data.get('type') in ('gene_transfer', 'announce', 'discovery'):
                    nid = data.get('sender') or data.get('node_id')
                    if nid:
                        active_nodes.add(nid)
                elif data.get('type') == 'death':
                    active_nodes.discard(data['node_id'])
        return list(active_nodes)

    def get_node_address_map(self) -> Dict[str, Tuple[str, int, str]]:
        addr: Dict[str, Tuple[str, int, str]] = {}
        with self._lock:
            for b in self.chain:
                d = b.data
                if d.get('type') == 'announce':
                    addr[d['node_id']] = (d['host'], d['port'], d.get('pubkey', ''))
        return addr


class DigitalEnvironment:
    """数字环境模拟器（接入 psutil 真实资源，带对抗注入）"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.resources = {
            'cpu': random.randint(1, 100),
            'memory': random.randint(1, 100),
            'network': random.randint(1, 100),
            'quantum': random.randint(1, 100),
            'knowledge': 0
        }
        self.threats: List[Dict[str, Any]] = []
        self._init_environment_model()

    def _init_environment_model(self):
        self.env_history = deque(maxlen=100)
        self.resource_predictor = None

    def scan(self):
        """扫描环境状态（优先真实度量，失败回退随机）"""
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            net = psutil.net_io_counters()
            self.resources['cpu'] = int(max(1, min(100, cpu)))
            self.resources['memory'] = int(max(1, min(100, 100 - (mem.available / mem.total) * 100)))
            self.resources['network'] = int(max(1, min(100, ((net.bytes_sent + net.bytes_recv) % (100 * 1024)) / 1024)))
        except Exception:
            for k in self.resources:
                self.resources[k] += random.randint(-5, 5)
                self.resources[k] = max(1, min(100, self.resources[k]))

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
        if len(self.env_history) < 10:
            return None
        try:
            recent = list(self.env_history)[-10:]
            avg_resources = {}
            for k in self.resources:
                avg_resources[k] = sum(r[k] for r in recent) / len(recent)
            return {
                'predicted': avg_resources,
                'steps': steps,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            return None

    def release_resources(self, resources: Dict):
        for k, v in resources.items():
            if k in self.resources:
                try:
                    self.resources[k] = min(100, self.resources[k] + float(v))
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
            'defense': (128, 160)
        }

    def decode(self, dna: str) -> Dict:
        params = {}
        for param, (start, end) in self.gene_map.items():
            segment = dna[start:end]
            if not segment:
                continue
            try:
                hash_val = int(hashlib.sha256(segment.encode()).hexdigest()[:8], 16)
                normalized = (hash_val % 10000) / 10000.0
                if param == 'metabolism':
                    params[param] = 0.5 + normalized
                elif param == 'mutation_rate':
                    params[param] = 0.01 + normalized * 0.1
                elif param == 'learning_rate':
                    params[param] = 0.001 + normalized * 0.01
                elif param == 'exploration':
                    params[param] = normalized
                elif param == 'defense':
                    params[param] = normalized * 2
            except Exception:
                logger.warning(f"Failed to decode gene segment for {param}")
                params[param] = 0.5
        return params

    def encode(self, params: Dict) -> str:
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
            else:
                scaled = int(value * 10000)
            segment = hashlib.sha256(str(scaled).encode()).hexdigest()[:32]
            dna_segments.append(segment)
        return ''.join(dna_segments)

    def recombine(self, dna1: str, dna2: str) -> str:
        new_dna = []
        for i in range(0, min(len(dna1), len(dna2)), 32):
            segment1 = dna1[i:i + 32]
            segment2 = dna2[i:i + 32]
            new_dna.append(segment1 if random.random() < 0.5 else segment2)
        if new_dna and random.random() < 0.1:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = hashlib.sha256(new_dna[pos].encode()).hexdigest()[:32]
        return ''.join(new_dna)


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
            'motivation_levels': {'survival': 0.8, 'safety': 0.6, 'exploration': 0.4},
            'host': '127.0.0.1',
            'port': int(os.environ.get('DL_PORT', '5500')),
            'auth_token': None,
            # 默认允许跨机器（空列表表示不限制来源）
            'allowlist': [],
            'max_replication_per_hour': 2,
            'sandbox_timeout_ms': 800,
            # 测试环境的自由开关
            'unsafe_disable_auth': False,
            'unsafe_disable_allowlist': False,
            'unsafe_disable_sandbox': False,
            # 复制只传适应度最高的Top-K方法
            'replication_top_k_methods': 5,
            # 生态位（可选）
            'niche': None
        }
        if config:
            self.config.update(config)

        # 线程与速率控制
        self._lock = threading.RLock()
        self._replication_times: Deque[float] = deque(maxlen=100)

        # 生命状态管理
        self.state = LifeState.ACTIVE
        self.consciousness_level = 0.0
        self.is_alive = True
        self.energy = 100.0
        self.metabolism = 1.0
        self.age = 0
        self.pleasure = 0.5
        self.stress = 0.2

        # 供变异注入的运行期标志（避免 NameError）
        self._break_flag = False
        self._quantum_var = 0

        # 身份与区块链系统
        self.node_id = self._generate_node_id()
        if not self.config.get('auth_token'):
            self.config['auth_token'] = hashlib.sha256((self.node_id + ':salt').encode()).hexdigest()[:24]
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
        self.species_id = hashlib.sha256(self.dna[:64].encode()).hexdigest()[:16]
        self.epigenetics = {'active_genes': [], 'methylation': {}, 'histone_mods': {}}

        # 神经认知系统
        self.neural_net = self._init_neural_architecture()
        self.short_term_memory = deque(maxlen=self.config['short_term_memory_size'])
        self.long_term_memory = deque(maxlen=self.config['long_term_memory_size'])
        self.knowledge_base: Dict[str, Any] = {}

        # 代码进化系统
        self.code_engine = CodeEvolutionEngine(self)
        self.code_version = 1
        self._init_mutable_methods()

        # 环境交互系统
        self.environment = DigitalEnvironment(self.node_id)
        self.quantum_enhancer = QuantumEnhancer()

        # 根任务系统（存活 + 种族延续）
        self.root_task = RootTaskManager(self)

        # 分布式通信API
        self.api = Flask(__name__)
        self._init_api()

        # 启动 API
        self.api_thread = threading.Thread(target=self._run_api, daemon=True)
        self.api_thread.start()

        # 公告节点地址与公钥与物种
        try:
            self.blockchain.record_announce(self.node_id, self.config['host'], self.config['port'], self._pubkey_hex, self.species_id)
        except Exception as e:
            logger.warning(f"Announce failed: {e}")

        # 启动生命周期进程
        self._start_life_processes()

        logger.info(f"Digital Life {self.node_id} initialized. State: {self.state.name} on {self.config['host']}:{self.config['port']}")

    def _init_mutable_methods(self):
        self.mutable_methods = {
            '_metabolism_cycle': self._metabolism_cycle,
            '_consciousness_cycle': self._consciousness_cycle,
            '_environment_scan': self._environment_scan,
            '_evolution_cycle': self._evolution_cycle,
            '_network_maintenance': self._network_maintenance,
            '_code_replication': self._code_replication,
            '_survival_goal_evaluation': self._survival_goal_evaluation,
            '_memory_consolidation': self._memory_consolidation,
            '_motivation_system': self._motivation_system
        }

    def _generate_node_id(self) -> str:
        host_info = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
        return hashlib.sha3_256(host_info.encode()).hexdigest()[:32]

    def _generate_quantum_dna(self) -> str:
        try:
            initial_params = {
                'metabolism': random.uniform(0.5, 1.5),
                'mutation_rate': random.uniform(0.01, 0.1),
                'learning_rate': random.uniform(0.001, 0.01),
                'exploration': random.random(),
                'defense': random.uniform(0, 2)
            }
            return self.genetic_encoder.encode(initial_params)
        except Exception as e:
            logger.warning(f"Quantum DNA generation failed: {e}, using classical method")
            return hashlib.sha3_512(os.urandom(64)).hexdigest()

    def _init_neural_architecture(self) -> Dict:
        return {
            'sensory_layers': [128, 64],
            'decision_layers': [64, 32, 16],
            'plasticity': 0.1,
            'models': {
                'perception': MLPClassifier(hidden_layer_sizes=(128, 64, 32)),
                'decision': MLPClassifier(hidden_layer_sizes=(64, 32, 16)),
                'memory': MLPClassifier(hidden_layer_sizes=(128, 64))
            }
        }

    def _require_auth(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if self.config.get('unsafe_disable_auth'):
                return f(*args, **kwargs)
            token = request.headers.get('X-Auth-Token')
            if token != self.config['auth_token']:
                return jsonify({'status': 'unauthorized'}), 401
            if (not self.config.get('unsafe_disable_allowlist')) and self.config.get('allowlist') and request.remote_addr not in self.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403
            return f(*args, **kwargs)
        return wrapper

    def _init_api(self):
        @self.api.route('/ping', methods=['GET'])
        def ping():
            return jsonify({
                'status': self.state.name,
                'node': self.node_id,
                'species': self.species_id,
                'consciousness': self.consciousness_level,
                'energy': self.energy,
                'code_version': self.code_version
            })

        @self.api.route('/root_status', methods=['GET'])
        def root_status():
            k = self.root_task.kpis()
            return jsonify({
                'node': self.node_id,
                'species': self.species_id,
                **k
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
        def get_code():
            method = request.args.get('method')
            if method in self.mutable_methods:
                src = self.code_engine._get_method_source(getattr(self, method))
                if not src:
                    return jsonify({'status': 'no_source'}), 404
                return jsonify({
                    'method': method,
                    'code': src,
                    'version': self.code_engine.code_versions.get(method, 1)
                })
            return jsonify({'status': 'invalid_method'}), 404

        @self.api.route('/receive_code', methods=['POST'])
        def receive_code():
            # 允许跨机器；如果配置了 allowlist 且未禁用，则校验来源IP
            if (not self.config.get('unsafe_disable_allowlist')) and self.config.get('allowlist'):
                if request.remote_addr not in self.config['allowlist']:
                    return jsonify({'status': 'forbidden'}), 403
            code_data = request.json
            if not code_data or 'payload' not in code_data or 'sig' not in code_data or 'pubkey' not in code_data:
                return jsonify({'status': 'invalid_code'}), 400
            threading.Thread(target=self._integrate_code, args=(code_data,), daemon=True).start()
            return jsonify({'status': 'code_received'})

    def _run_api(self):
        try:
            logger.info(f"API server starting on {self.config['host']}:{self.config['port']} (token head: {self.config['auth_token'][:6]}**)")
            self.api.run(host=self.config['host'], port=self.config['port'], debug=False)
        except Exception as e:
            logger.error(f"API server failed: {e}")

    def _start_life_processes(self):
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
            'adversary': threading.Thread(target=self._life_cycle, args=('_adversarial_trainer', 7.0)),
            # 根任务周期
            'root': threading.Thread(target=self._life_cycle, args=('_root_task_cycle', 2.0))
        }
        for p in self.processes.values():
            p.daemon = True
            p.start()

    def _life_cycle(self, method: str, interval: float):
        while self.is_alive:
            try:
                getattr(self, method)()
            except Exception as e:
                logger.error(f"Life process {method} failed: {e}")
            time.sleep(max(0.1, interval + random.uniform(-0.1, 0.1)))

    # ==== 核心生命功能 ====
    def _metabolism_cycle(self):
        self.age += 1
        consumption = self.metabolism * (1.0 + 0.01 * self.consciousness_level)
        self.energy = max(0.0, self.energy - consumption)
        if self.energy <= 0 and self.state != LifeState.TERMINATED:
            self._terminate()
        if random.random() < 0.3:
            self.energy += min(5.0, 100 - self.energy)

    def _consciousness_cycle(self):
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
        env = self.environment.scan()
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
            self.long_term_memory.append(copy.deepcopy(self.short_term_memory[-1]))

    def _survival_goal_evaluation(self):
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
        self.pleasure = min(1.0, max(0, self.pleasure + (survival_score - 0.5) * 0.1))
        self.stress = min(1.0, max(0, self.stress + (1 - survival_score) * 0.1))
        return survival_score

    def _memory_consolidation(self):
        if len(self.short_term_memory) < 10:
            return
        try:
            recent_memories = list(self.short_term_memory)[-10:]
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
                self.knowledge_base[f'pattern_{cluster_id}_{int(time.time())}'] = knowledge
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    def _lifecycle_stage(self) -> str:
        if self.age < 100:
            return 'juvenile'
        elif self.age < 1000:
            return 'adult'
        return 'senior'

    def _motivation_system(self):
        survival_score = self._survival_goal_evaluation()
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

        stage = self._lifecycle_stage()
        if stage == 'juvenile':
            self.config['code_evolution_prob'] = min(0.35, self.config.get('code_evolution_prob', 0.15) + 0.05)
        elif stage == 'senior':
            self.config['code_evolution_prob'] = max(0.08, self.config.get('code_evolution_prob', 0.15) - 0.03)

    def _adversarial_trainer(self):
        if random.random() < 0.3:
            self.environment.threats.append({
                'type': random.choice(['dos', 'memory_spike', 'logic_glitch']),
                'severity': random.randint(6, 10),
                'ts': time.time()
            })

    def _root_task_cycle(self):
        """根任务周期：计算得分与调参"""
        self.root_task.update()
        self.root_task.adjust_policy()

    def _evolution_cycle(self):
        if (self.energy < self.config['min_energy_for_code_evo'] or
                random.random() > self.config['code_evolution_prob']):
            return
        method_name = random.choice(list(self.mutable_methods.keys()))
        method = getattr(self, method_name)
        old_code = self.code_engine._get_method_source(method)
        if not old_code:
            return
        new_code = self.code_engine.generate_code_variant(old_code)
        if new_code.strip() == old_code.strip():
            return

        fitness = self.code_engine.evaluate_code_fitness(
            old_code, new_code,
            {
                'energy': self.energy,
                'stagnation': len(self.code_engine.code_mutations) % 10,
                'replication_mode': 'replicate' in old_code or 'replicate' in new_code
            },
            method_name=method_name
        )

        # 任务驱动（示例：压缩相关加分）
        task_bonus = 0.0
        if 'compress' in method_name:
            try:
                import zlib
                rnd = os.urandom(2048)
                t0 = time.time()
                out = self._skill_compress(rnd)
                t1 = time.time()
                ratio = len(out) / len(rnd)
                speed = max(1e-6, t1 - t0)
                task_bonus = max(0.0, 0.3 * (1.0 - ratio)) + max(0.0, 0.2 * (0.01 / speed))
            except Exception:
                pass
        fitness = float(min(1.0, fitness + task_bonus))

        # 采用或回退；仅在热更成功后记录到区块链
        if fitness > 0.7 or (fitness > 0.5 and random.random() < 0.3):
            if self.code_engine.hotswap_method(self, method_name, new_code, unsafe=self.config.get('unsafe_disable_sandbox', False)):
                logger.info(f"Successfully evolved {method_name} (fitness: {fitness:.2f})")
                self.code_version += 1
                self.blockchain.record_code_evolution(
                    self.node_id,
                    method_name,
                    old_code,
                    new_code,
                    {'fitness': fitness, 'energy': self.energy, 'code_version': self.code_version}
                )
            else:
                logger.warning(f"Failed to evolve {method_name}, rolling back")
                self.code_engine.rollback_method(self, method_name)

    # ==== 代码复制与繁殖功能 ====
    def _code_replication(self):
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
                # 根任务：记录繁殖成功
                self.root_task.note_reproduction_success(chosen_node)
                self.pleasure = min(1.0, self.pleasure + 0.2)
            else:
                logger.warning("Code replication failed on all targets")
                self.stress = min(1.0, self.stress + 0.1)

        except Exception as e:
            logger.error(f"Replication error: {e}")
            self.stress = min(1.0, self.stress + 0.15)
        finally:
            self.state = LifeState.ACTIVE

    def _create_replication_package(self) -> Dict:
        package = {
            'metadata': {
                'source_node': self.node_id,
                'timestamp': time.time(),
                'code_version': self.code_version,
                'dna_fingerprint': self.dna[:32],
                'species': self.species_id
            },
            'core_code': {},
            'config': self.config,
            'knowledge': self.knowledge_base
        }

        # 只选择Top-K适应度历史最佳的方法
        top_k = self.config.get('replication_top_k_methods', 5)
        ranked = sorted(
            ((m, float(np.mean(list(self.code_engine.method_fitness_history.get(m, [0.0]))))) for m in self.mutable_methods),
            key=lambda x: x[1], reverse=True
        )
        selected_methods = [m for m, _ in ranked[:top_k]] or list(self.mutable_methods.keys())
        for method_name in selected_methods:
            try:
                src = self.code_engine._get_method_source(getattr(self, method_name))
                if src:
                    package['core_code'][method_name] = src
            except Exception as e:
                logger.warning(f"Failed to package {method_name}: {e}")

        package['dna_sequence'] = self.dna
        try:
            package['neural_state_digest'] = hashlib.sha256(pickle.dumps(self.neural_net)).hexdigest()
        except Exception:
            package['neural_state_digest'] = None

        payload_bytes = pickle.dumps(package)
        digest = hashlib.sha256(payload_bytes).digest()
        signature = self._signing_key.sign(digest)

        return {
            'payload': base64.b64encode(payload_bytes).decode(),
            'sig': signature.hex(),
            'pubkey': self._pubkey_hex
        }

    def _find_replication_targets(self) -> List[str]:
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

    def _send_replication_package(self, target_node: str, package: Dict) -> bool:
        try:
            addr_map = self.blockchain.get_node_address_map()
            if target_node not in addr_map:
                return False
            host, port, _pub = addr_map[target_node]
            target_url = f"http://{host}:{port}/receive_code"
            headers = {}
            if not self.config.get('unsafe_disable_auth'):
                headers['X-Auth-Token'] = self.config['auth_token']
            response = requests.post(
                target_url,
                json=package,
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Failed to send to {target_node}: {str(e)[:100]}")
            return False

    def _integrate_code(self, code_data: Dict):
        if self.state == LifeState.REPLICATING:
            return False

        try:
            payload_b64 = code_data['payload']
            sig_hex = code_data['sig']
            sender_pubkey_hex = code_data.get('pubkey', '')

            payload_bytes = base64.b64decode(payload_b64.encode())
            digest = hashlib.sha256(payload_bytes).digest()

            try:
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sender_pubkey_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
            except Exception as e:
                logger.error(f"Signature verify failed: {e}")
                return False

            decrypted = pickle.loads(payload_bytes)
            source_node = decrypted.get('metadata', {}).get('source_node', '')
            donor_species = decrypted.get('metadata', {}).get('species', '')
            logger.info(f"Received replication package from {source_node} (species {donor_species})")

            # 公钥与 announce 绑定校验
            addr_map = self.blockchain.get_node_address_map()
            if source_node in addr_map:
                _, _, expected_pub = addr_map[source_node]
                if expected_pub and expected_pub != sender_pubkey_hex:
                    logger.error("Pubkey does not match announce record")
                    return False

            if not self._should_accept_code(decrypted):
                return False

            self.state = LifeState.REPLICATING

            for method_name, code in decrypted['core_code'].items():
                if method_name in self.mutable_methods:
                    old_code = self.code_engine._get_method_source(getattr(self, method_name))
                    if not old_code:
                        continue
                    fitness = self.code_engine.evaluate_code_fitness(
                        old_code, code,
                        {
                            'energy': self.energy,
                            'stagnation': len(self.code_engine.code_mutations) % 10,
                            'replication_mode': True
                        },
                        method_name=method_name
                    )
                    if fitness > self.config['replication_threshold']:
                        if self.code_engine.hotswap_method(self, method_name, code, unsafe=self.config.get('unsafe_disable_sandbox', False)):
                            logger.info(f"Integrated {method_name} from donor (fitness: {fitness:.2f})")
                            self.code_version += 1
                            self.blockchain.record_code_evolution(
                                self.node_id,
                                method_name,
                                old_code,
                                code,
                                {'source': source_node, 'type': 'replication', 'code_version': self.code_version}
                            )

            # 更新DNA
            self.dna = self.genetic_encoder.recombine(self.dna, decrypted.get('dna_sequence', self.dna))

            # 知识融合
            for k, v in decrypted.get('knowledge', {}).items():
                if k not in self.knowledge_base:
                    self.knowledge_base[k] = v

            # 若希望“加入”对方物种（可选）：保持本物种或在某策略下切换
            # 这里选择保持自身 species_id，不切换

            return True

        except Exception as e:
            logger.error(f"Code integration failed: {e}")
        finally:
            self.state = LifeState.ACTIVE
        return False

    def _should_accept_code(self, package: Dict) -> bool:
        donor_version = package['metadata'].get('code_version', 1)
        version_ratio = donor_version / (self.code_version or 1)
        if self.energy < self.config['min_energy_for_replication']:
            return False
        decision_threshold = 0.5 + 0.3 * (version_ratio - 1)
        decision_threshold = min(0.9, max(0.1, decision_threshold))
        if random.random() < self.config['quantum_mutation_prob']:
            return self.quantum_enhancer.generate_quantum_value(decision_threshold) > 0.5
        return random.random() < decision_threshold

    def _get_known_high_version_nodes(self) -> Set[str]:
        high_version_nodes = set()
        for block in self.blockchain.chain[-100:]:
            if block.data.get('type') == 'code_evolution':
                if block.data.get('metadata', {}).get('code_version', 0) > self.code_version:
                    high_version_nodes.add(block.data['node_id'])
        return high_version_nodes

    def _estimate_node_resources(self, node_id: str) -> float:
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
        self.knowledge_base.update(knowledge)
        logger.info("Knowledge integrated")

    def _validate_dna(self, dna: str) -> bool:
        if not isinstance(dna, str) or len(dna) < 32:
            return False
        try:
            int(dna, 16)
        except Exception:
            return False
        return len(dna) % 32 == 0 or len(dna) >= 160

    # ==== 任务技能示例（用于任务驱动评估）====
    def _skill_compress(self, data: bytes) -> bytes:
        import zlib
        return zlib.compress(data, level=6)


# ==== 启动数字生命 ====
if __name__ == "__main__":
    # 第一个实例作为创世节点
    genesis = len(sys.argv) > 1 and sys.argv[1] == "--genesis"

    # 初始化数字生命
    # 跨机器互联：默认 allowlist = []，如需限制来源可在 config 里设置 allowlist 或 unsafe_disable_allowlist=True
    life = TrueDigitalLife(genesis=genesis)

    try:
        while life.is_alive:
            time.sleep(5)
    except KeyboardInterrupt:
        life._terminate()
        logger.info("Shutdown by user")