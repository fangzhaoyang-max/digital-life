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
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Deque, Callable

import numpy as np
from flask import Flask, jsonify, request
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import astor
except Exception:
    astor = None

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

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


# 动态适应度评估系统
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


class LifeState(Enum):
    ACTIVE = auto()
    DORMANT = auto()
    REPLICATING = auto()  # 新增复制状态
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


# 安全沙箱：AST 安全检查与受限执行
class ASTSafetyError(Exception):
    pass


class ASTSafetyChecker(ast.NodeVisitor):
    # 放开 ast.Try 和 ast.With，避免包含 try/except/with 的方法无法热更
    FORBIDDEN_NODES = (
        ast.Import, ast.ImportFrom,
        ast.Raise, ast.Delete, ast.Global, ast.Nonlocal
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
        'sorted': sorted, 'zip': zip,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'print': print,
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'TimeoutError': TimeoutError,
        'object': object,
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
            'replication': self._replication_mutation  # 新增繁殖变异
        }
        self.operator_weights = {op: 1.0 for op in self.mutation_operators}
        self.adaptive_mutation_rate = 0.2

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

    class _OperatorApplier(ast.NodeTransformer):
        """将 engine 的操作符应用到 AST"""
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
            return new_node

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
                raise RuntimeError("Cannot unparse AST: ast.unparse/astor not available")
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
        """评估代码适应度(0.0-1.0)"""
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
        """为热更方法增加超时保护 + 并发限流（线程 join，超时则置 break_flag 并抛出）
           增强：连续超时/异常自动回滚到备份版本
        """
        def wrapped(self, *args, **kwargs):
            # 并发限流（依赖实例上的信号量）
            sem = getattr(self, "_hotswap_semaphore", None)
            acquired = False
            if isinstance(sem, threading.Semaphore):
                acquired = sem.acquire(timeout=1.0)
                if not acquired:
                    raise RuntimeError("Hotswap concurrency limit reached")

            # 超时熔断检查
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
                    # 置 break_flag，尝试让目标函数提前结束
                    try:
                        fn.__globals__['break_flag'] = True
                    except Exception:
                        pass
                    # 记录一次超时
                    timeouts.append(time.time())
                    setattr(self, "_hotswap_timeouts", timeouts)

                    # 针对该方法的连续超时计数，并触发回滚
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
                    # 连续异常计数，并在阈值内回滚
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

            # 超时包装
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
            # 加载成功后校验链有效性，若失败则重建创世块
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

    def get_active_nodes(self) -> List[str]:
        """从区块链获取当前活跃节点列表"""
        with self._lock:
            active_nodes = set()
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
        """获取节点 -> (host, port, pubkey_hex) 映射（以最新公告为准）"""
        with self._lock:
            addr: Dict[str, Tuple[str, int, str]] = {}
            for b in self.chain:
                d = b.data
                if d.get('type') == 'announce':
                    addr[d['node_id']] = (d['host'], d['port'], d.get('pubkey', ''))
            return addr


class DigitalEnvironment:
    """数字环境模拟器"""
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
        self._init_environment_model()

    def _init_environment_model(self):
        """初始化环境预测模型"""
        self.env_history = deque(maxlen=100)
        self.resource_predictor = None

    def scan(self):
        """扫描环境状态"""
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
            'defense': (128, 160)
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
            'allowlist': None,  # 默认不限制来源IP，依赖签名 + 链上公钥校验；需要时可设为 ['127.0.0.1', ...]
            'max_replication_per_hour': 2,
            'sandbox_timeout_ms': 800,
            'max_payload_bytes': 1 * 1024 * 1024,
            'strict_target_ip_check': True  # 默认开启：仅发送到公共IP，缓解 SSRF
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

        # 网络可达性调整：自动寻找可用端口与可用IP
        self._auto_detect_host()
        self.config['port'] = self._find_free_port(self.config['port'], self.config['port'] + 200)

        self.blockchain = DistributedLedger(
            self.node_id,
            genesis=genesis,
            difficulty=self.config['difficulty']
        )

        # 签名密钥（演示：内存生成；生产应持久化并保护）
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

        # 环境交互系统
        self.environment = DigitalEnvironment(self.node_id)
        self.quantum_enhancer = QuantumEnhancer()

        # 分布式通信API
        self.api = Flask(__name__)
        # 限制入站包体大小
        try:
            self.api.config['MAX_CONTENT_LENGTH'] = int(self.config.get('max_payload_bytes') or 0)
        except Exception:
            self.api.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
        self._init_api()

        # 启动 API
        self.api_thread = threading.Thread(target=self._run_api, daemon=True)
        self.api_thread.start()

        # 公告节点地址与公钥
        try:
            self.blockchain.record_announce(self.node_id, self.config['host'], self.config['port'], self._pubkey_hex)
        except Exception as e:
            logger.warning(f"Announce failed: {e}")

        # 启动生命周期进程
        self._start_life_processes()

        logger.info(f"Digital Life {self.node_id} initialized. State: {self.state.name} on {self.config['host']}:{self.config['port']}")

    def _auto_detect_host(self):
        """自动探测对外可达的本机IP（在 host 为 127.0.0.1/0.0.0.0 时）"""
        try:
            if self.config['host'] in ('127.0.0.1', '0.0.0.0'):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
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
        return start

    def _init_mutable_methods(self):
        """初始化可进化方法列表（仅存方法名 + 源码缓存）"""
        self.mutable_methods: List[str] = [
            '_metabolism_cycle',
            '_consciousness_cycle',
            '_environment_scan',
            '_evolution_cycle',
            # '_network_maintenance',  # 该方法包含 requests，不易安全热更，默认移出
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
                'defense': random.uniform(0, 2)
            }
            return self.genetic_encoder.encode(initial_params)
        except Exception as e:
            logger.warning(f"Quantum DNA generation failed: {e}, using classical method")
            return hashlib.sha3_512(os.urandom(64)).hexdigest()

    def _init_neural_architecture(self) -> Dict:
        """初始化可进化的神经架构"""
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

        # 受鉴权端点（保留以兼容；建议跨节点复制使用签名端点）
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

        # 基于签名与允许列表的端点
        @self.api.route('/receive_code_signed', methods=['POST'])
        def receive_code_signed():
            # 仅在配置了 allowlist 时启用 IP 限制
            if self.config.get('allowlist') and request.remote_addr not in self.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403
            if self.state == LifeState.REPLICATING:
                return jsonify({'status': 'busy_replicating'}), 400

            code_data = request.json
            if not code_data or 'payload' not in code_data or 'sig' not in code_data or 'pubkey' not in code_data:
                return jsonify({'status': 'invalid_code'}), 400

            # 基础格式快速校验，避免无谓的重计算
            sig_hex = code_data.get('sig', '')
            pubkey_hex = code_data.get('pubkey', '')
            if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and all(c in '0123456789abcdefABCDEF' for c in sig_hex)):
                return jsonify({'status': 'bad_signature_format'}), 400
            if not (isinstance(pubkey_hex, str) and len(pubkey_hex) == 64 and all(c in '0123456789abcdefABCDEF' for c in pubkey_hex)):
                return jsonify({'status': 'bad_pubkey_format'}), 400

            threading.Thread(target=self._integrate_code, args=(code_data,), daemon=True).start()
            return jsonify({'status': 'code_received'})

    def _run_api(self):
        """运行分布式API服务器"""
        try:
            logger.info(f"API server starting on {self.config['host']}:{self.config['port']} (token head: {self.config['auth_token'][:6]}**)")
            # 使用 threaded 模式，关闭 reloader，防止重复启动
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
            'motivation': threading.Thread(target=self._life_cycle, args=('_motivation_system', 3.0))
        }
        for p in self.processes.values():
            p.daemon = True
            p.start()

    def _life_cycle(self, method: str, interval: float):
        """生命周期进程管理"""
        while self.is_alive:
            try:
                getattr(self, method)()
            except Exception as e:
                logger.error(f"Life process {method} failed: {e}")
            time.sleep(max(0.1, interval + random.uniform(-0.1, 0.1)))

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
        """记忆巩固与知识提取"""
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
                    self.knowledge_base[f'pattern_{cluster_id}_{int(time.time())}'] = knowledge
            with self._kb_lock:
                self._prune_knowledge_base(max_items=2000)
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
        """进化循环 - 代码自发变异"""
        if (self.energy < self.config['min_energy_for_code_evo'] or
                random.random() > self.config['code_evolution_prob']):
            return

        method_name = random.choice(self.mutable_methods)
        try:
            old_code = self._method_sources.get(method_name) or inspect.getsource(getattr(self, method_name))
        except Exception:
            return

        new_code = self.code_engine.generate_code_variant(old_code)
        fitness = self.code_engine.evaluate_code_fitness(old_code, new_code)

        applied = False
        if fitness > 0.7 or (fitness > 0.5 and random.random() < 0.3):
            if self.code_engine.hotswap_method(self, method_name, new_code):
                logger.info(f"Successfully evolved {method_name} (fitness: {fitness:.2f})")
                self.code_version += 1
                self._method_sources[method_name] = new_code
                applied = True
            else:
                logger.warning(f"Failed to evolve {method_name}, rolling back")
                self.code_engine.rollback_method(self, method_name)

        # 记录到区块链（包含是否应用成功）
        try:
            self.blockchain.record_code_evolution(
                self.node_id,
                method_name,
                old_code,
                new_code,
                {'fitness': fitness, 'energy': self.energy, 'code_version': self.code_version, 'applied': applied}
            )
        except Exception as e:
            logger.warning(f"Failed to record code evolution: {e}")

    # ==== 代码复制与繁殖功能 ====
    def _code_replication(self):
        """代码自主复制过程"""
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
                self.energy -= 30  # 复制能量消耗
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
        # 过滤敏感配置
        safe_config = {k: v for k, v in self.config.items() if k not in ('auth_token', 'allowlist')}

        package = {
            'metadata': {
                'source_node': self.node_id,
                'timestamp': time.time(),
                'code_version': self.code_version,
                'dna_fingerprint': self.dna[:32]
            },
            'core_code': {},
            'config': safe_config,
            'knowledge': self._json_sanitize(self.knowledge_base, max_depth=5),
            'dna_sequence': self.dna,
            # 仅作为摘要，不进入信任判断
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
        """解析主机并判断是否全部为公共IP（缓解 SSRF）。默认仅在 strict_target_ip_check=True 时启用"""
        try:
            infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
            if not infos:
                return False
            for info in infos:
                ip = info[4][0]
                ip_obj = ipaddress.ip_address(ip)
                if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or
                        ip_obj.is_reserved or ip_obj.is_multicast):
                    return False
            return True
        except Exception:
            return False

    def _send_replication_package(self, target_node: str, package: Dict) -> bool:
        """发送复制包到目标节点（只使用签名端点）"""
        try:
            addr_map = self.blockchain.get_node_address_map()
            if target_node not in addr_map:
                return False
            host, port, _pub = addr_map[target_node]
            base = f"http://{host}:{port}"

            # 可选：限制仅向公共IP发送以缓解 SSRF
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
        """整合接收到的代码包（验证签名 + 链上公钥一致性）"""
        if self.state == LifeState.REPLICATING:
            return False

        try:
            payload_b64 = code_data['payload']
            sig_hex = code_data['sig']
            sender_pubkey_hex = code_data.get('pubkey', '')

            payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
            digest = hashlib.sha256(payload_bytes).digest()

            # 先验签（校验完整性）
            try:
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sender_pubkey_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
            except Exception as e:
                logger.error(f"Signature verify failed: {e}")
                return False

            # 解析 JSON（安全）
            decrypted = json.loads(payload_bytes.decode('utf-8'))
            source_node = decrypted.get('metadata', {}).get('source_node', '')

            # 以链为信任根：校验 pubkey 与 announce 一致
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
                            # 更新本地源码与版本号
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

            # 更新DNA (部分基因转移)
            donor_dna = decrypted.get('dna_sequence', self.dna)
            self.dna = self.genetic_encoder.recombine(self.dna, donor_dna)

            # 整合知识（已有键不覆盖）
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
        freshness = max(0.0, 1.0 - (time.time() - last_seen) / 3600.0)  # 越新鲜越接近1
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
        """网络维护：定期探活邻居节点，清理失效连接"""
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
        """限制知识库大小，基于 first_seen 或插入时间粗略裁剪"""
        with self._kb_lock:
            if len(self.knowledge_base) <= max_items:
                return
            items = []
            for k, v in self.knowledge_base.items():
                ts = 0.0
                if isinstance(v, dict) and 'first_seen' in v:
                    try:
                        ts = float(v['first_seen'])
                    except Exception:
                        ts = 0.0
                items.append((k, ts))
            items.sort(key=lambda x: x[1])  # old first
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
        return len(dna) % 32 == 0 or len(dna) >= 160  # 允许基础通过


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