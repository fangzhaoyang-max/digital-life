"""
Fitness evaluation components for the Digital Life system
"""

import ast
import difflib
import random
import sys
import textwrap
import threading
import time
from typing import Dict, List, Callable, Any, Optional, Set, Tuple

import numpy as np

from ..imports import _hyp, _st, _given, _hsettings, torch, nn, fx, astor, logger
from ..utils.common import _clamp
from .safe_exec import SafeExec, ASTSafetyError


class DynamicFitnessEvaluator:
    """Dynamic multi-metric fitness evaluator"""

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
        mode = self._select_evaluation_mode(context)
        weights = self.adaptive_weights[mode]

        scores = {
            'functionality': self._functionality_score(original, mutated),
            'novelty': self._novelty_score(original, mutated),
            'complexity': self._complexity_score(mutated),
            'energy_efficiency': self._energy_efficiency_score(mutated),
            'replicability': self._replicability_score(mutated)
        }

        return sum(scores[k] * weights.get(k, 0) for k in scores) / max(1e-9, sum(weights.values()))

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
            src = textwrap.dedent(mutated).lstrip()
            compile(src, '<string>', 'exec')
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
            src = textwrap.dedent(code).lstrip()
            tree = ast.parse(src)
            complexity = len(list(ast.walk(tree))) / 100
            return min(1.0, complexity)
        except Exception:
            return 0.5

    def _energy_efficiency_score(self, code: str) -> float:
        lines = code.splitlines()
        return 1.0 / (1 + len(lines) / 10)

    def _replicability_score(self, code: str) -> float:
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


class ParetoTools:
    """Pareto ranking utilities"""

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
    """Automated correctness harness for evolved methods"""

    def __init__(self, owner):
        self.owner = owner
        self.tests: Dict[str, List[Callable[[Any, Callable], bool]]] = {}
        self.test_value: Dict[str, List[float]] = {}
        self.max_tests_per_method = 16

    def _ensure_tests(self, method_name: str):
        if method_name in self.tests:
            return
        gens: List[Callable[[Any, Callable], bool]] = []

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
                return isinstance(s, (int, float)) and not (s != s)

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
            snap = {
                'energy': instance.energy,
                'age': instance.age,
                'state': instance.state,
                'pleasure': instance.pleasure,
                'stress': instance.stress,
                'env_res': instance.environment.resources.copy(),
                'env_thr': instance.environment.threats.copy(),
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
    """Multi-objective fitness evaluation"""

    def __init__(self, test_harness: CorrectnessHarness):
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
            has_rep = False
            has_emergent_call = False
            has_local_def = False

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Name) and 'replicate' in f.id:
                        has_rep = True
                    if isinstance(f, ast.Attribute):
                        if isinstance(f.value, ast.Name) and f.value.id == 'self':
                            if isinstance(f.attr, str) and (f.attr.startswith('m_') or f.attr.startswith('em_') or f.attr.startswith('f_')):
                                has_emergent_call = True
                if isinstance(node, ast.FunctionDef):
                    for ch in node.body:
                        if isinstance(ch, ast.FunctionDef):
                            has_local_def = True

            return 0.8 if (has_rep or has_emergent_call or has_local_def) else 0.2
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
