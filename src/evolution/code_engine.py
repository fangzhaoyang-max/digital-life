"""Core code evolution engine for Digital Life."""
from __future__ import annotations

import ast
import inspect
import random
import textwrap
import threading
import time
import types
import uuid
from collections import deque
from typing import Any, Dict, Optional, Sequence

from ..imports import astor, logger
from ..utils.common import _clamp
from .fitness import (
    CorrectnessHarness,
    DynamicFitnessEvaluator,
    MultiObjectiveFitness,
    ParetoTools,
)
from .quantum_enhancer import QuantumEnhancer
from .safe_exec import SafeExec


class CodeEvolutionEngine:
    """Coordinates code mutation, selection, and method hot-swapping."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self.quantum = QuantumEnhancer()
        self.fitness = DynamicFitnessEvaluator()
        self.harness = CorrectnessHarness(owner)
        self.multi_objective = MultiObjectiveFitness(self.harness)

        self.operator_weights: Dict[str, float] = {}
        self.code_versions: Dict[str, int] = {}
        self.backup_methods: Dict[str, types.MethodType] = {}
        self.emergent_methods: set[str] = set()

        self._init_operator_bank()

    # ------------------------------------------------------------------
    # Operator management
    # ------------------------------------------------------------------
    def _init_operator_bank(self) -> None:
        self.operators = {
            "flip_control_flow": self._mutate_control_flow,
            "perturb_constant": self._mutate_constant,
            "jitter_assignment": self._mutate_assignment_literal,
            "noop": lambda node: node,
        }
        self.operator_weights = {name: 1.0 for name in self.operators}

    def _choose_operator(self) -> str:
        total = sum(self.operator_weights.values()) or 1.0
        roll = random.uniform(0.0, total)
        upto = 0.0
        for name, weight in self.operator_weights.items():
            upto += weight
            if upto >= roll:
                return name
        return random.choice(list(self.operators))

    def _apply_operator(self, node: ast.AST) -> ast.AST:
        choice = self._choose_operator()
        mutated = self.operators[choice](node)
        if mutated is not node:
            for name in self.operator_weights:
                if name == choice:
                    self.operator_weights[name] *= 1.05
                else:
                    self.operator_weights[name] *= 0.995
            average = (sum(self.operator_weights.values()) or 1.0) / len(self.operator_weights)
            for name in self.operator_weights:
                self.operator_weights[name] = _clamp(self.operator_weights[name] / average, 0.2, 5.0)
        return mutated

    # ------------------------------------------------------------------
    # Mutation primitives
    # ------------------------------------------------------------------
    def _mutate_control_flow(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.If):
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node

    def _mutate_constant(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            try:
                mutated = self.quantum.generate_quantum_value(node.value)
                return ast.Constant(value=float(mutated))
            except Exception:  # pragma: no cover - defensive
                return node
        return node

    def _mutate_assignment_literal(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            jitter = random.randint(-5, 5)
            node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=jitter))
        return node

    # ------------------------------------------------------------------
    # Variant generation & evaluation
    # ------------------------------------------------------------------
    def generate_code_variant(self, source: str) -> str:
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError:
            return source

        engine = self

        class _Mutator(ast.NodeTransformer):
            def generic_visit(self, node: ast.AST) -> ast.AST:
                node = super().generic_visit(node)
                return engine._apply_operator(node)

        mutated = _Mutator().visit(tree)
        ast.fix_missing_locations(mutated)

        if hasattr(ast, "unparse"):
            return textwrap.dedent(ast.unparse(mutated))
        if astor:
            return textwrap.dedent(astor.to_source(mutated))
        return source

    def evaluate_variants(self, method_name: str, original: str, variants: Sequence[str]) -> Dict[str, Any]:
        evaluations = []
        for candidate in variants:
            metrics = self.multi_objective.evaluate(self.owner, method_name, original, candidate)
            evaluations.append(metrics)

        maximize = {"correctness", "replicability"}
        minimize = {"energy_cost", "complexity"}

        fronts = ParetoTools.non_dominated_sort(evaluations, maximize, minimize)
        champion_front = fronts[0] if fronts else list(range(len(variants)))
        crowding = ParetoTools.crowding_distance(
            evaluations,
            champion_front,
            [(metric, True) for metric in maximize] + [(metric, False) for metric in minimize],
        )
        champion = max(
            champion_front,
            key=lambda idx: (evaluations[idx].get("correctness", 0.0), crowding.get(idx, 0.0)),
        )

        return {
            "objectives": evaluations,
            "champion_index": champion,
            "champion_code": variants[champion],
        }

    def evaluate_code_fitness(self, original: str, mutated: str) -> float:
        context = {
            "energy": getattr(self.owner, "energy", 1.0),
            "stagnation": len(getattr(self.owner, "code_engine_history", [])) % 10,
            "replication_mode": "replicate" in original or "replicate" in mutated,
        }
        return float(self.fitness.evaluate(original, mutated, context))

    # ------------------------------------------------------------------
    # Hot-swap support
    # ------------------------------------------------------------------
    def _wrap_with_timeout(self, fn, timeout_ms: int):
        def wrapped(self, *args, **kwargs):
            semaphore = getattr(self, "_hotswap_semaphore", None)
            acquired = False
            if isinstance(semaphore, threading.Semaphore):
                acquired = semaphore.acquire(timeout=1.0)
                if not acquired:
                    raise RuntimeError("Hotswap concurrency limit reached")

            timeouts: deque[float] = getattr(self, "_hotswap_timeouts", deque(maxlen=32))
            now = time.time()
            recent = [timestamp for timestamp in timeouts if now - timestamp < 60.0]
            if len(recent) >= 5:
                if acquired:
                    semaphore.release()
                raise RuntimeError("Hotswap temporarily suspended")

            result: Dict[str, Any] = {"value": None, "error": None}

            def runner() -> None:
                try:
                    result["value"] = fn(self, *args, **kwargs)
                except Exception as exc:  # pragma: no cover - sandbox foil
                    result["error"] = exc

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            thread.join(max(timeout_ms, 1) / 1000.0)

            try:
                if thread.is_alive():
                    try:
                        fn.__globals__["break_flag"] = True
                    except Exception:
                        pass
                    timeouts.append(time.time())
                    setattr(self, "_hotswap_timeouts", timeouts)
                    raise TimeoutError(f"Sandbox timeout: {fn.__name__}")

                if result["error"] is not None:
                    raise result["error"]

                return result["value"]
            finally:
                if acquired:
                    semaphore.release()

        wrapped.__name__ = fn.__name__
        return wrapped

    def hotswap_method(self, instance, method_name: str, new_code: str) -> bool:
        try:
            compiled = SafeExec.compile_and_load(
                new_code,
                method_name,
                filename=f"<mutation:{method_name}:{uuid.uuid4().hex[:8]}>",
                extra_globals={
                    "__quantum_var__": random.randint(0, 1),
                    "break_flag": False,
                    "time": time,
                },
            )
        except Exception as exc:
            logger.error("Failed to compile evolved method %s: %s", method_name, exc)
            return False

        if method_name not in self.backup_methods and hasattr(instance, method_name):
            current = getattr(instance, method_name)
            if isinstance(current, types.MethodType):
                self.backup_methods[method_name] = current

        timeout_ms = int(getattr(instance, "config", {}).get("sandbox_timeout_ms", 800))
        wrapped = self._wrap_with_timeout(compiled, timeout_ms)
        setattr(instance, method_name, types.MethodType(wrapped, instance))

        self.code_versions[method_name] = self.code_versions.get(method_name, 0) + 1
        if hasattr(instance, "_method_sources"):
            instance._method_sources[method_name] = new_code
        self.emergent_methods.add(method_name)
        return True

    def rollback_method(self, instance, method_name: str) -> bool:
        backup = self.backup_methods.get(method_name)
        if not backup:
            return False
        setattr(instance, method_name, backup)
        logger.info("Rolled back method %s to previous revision", method_name)
        return True

    # ------------------------------------------------------------------
    # Emergent method synthesis
    # ------------------------------------------------------------------
    def synthesize_and_attach_method(self, prefix: str = "m") -> Optional[str]:
        if not getattr(self.owner, "config", {}).get("allow_emergent_functions", True):
            return None

        name = f"{prefix}_{uuid.uuid4().hex[:8]}"
        while hasattr(self.owner, name):
            name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        template = f'''
def {name}(self, *args, **kwargs):
    """Auto-synthesized helper."""
    try:
        return {{
            "name": "{name}",
            "ts": time.time(),
            "energy": float(getattr(self, "energy", 0.0)),
            "state": getattr(getattr(self, "state", None), "name", None),
        }}
    except Exception as exc:
        return {{"ok": False, "error": str(exc)}}
'''

        code = textwrap.dedent(template)
        if self.hotswap_method(self.owner, name, code):
            if hasattr(self.owner, "mutable_methods"):
                self.owner.mutable_methods.append(name)
            return name
        return None

    # ------------------------------------------------------------------
    # Batch workflow
    # ------------------------------------------------------------------
    def evolve_method(self, method_name: str, population_size: int = 6) -> Optional[Dict[str, Any]]:
        try:
            source = self.owner._method_sources.get(method_name)
            if not source:
                source = inspect.getsource(getattr(self.owner, method_name))
        except (OSError, AttributeError):
            logger.debug("Unable to access source for method %s", method_name)
            return None

        population = [self.generate_code_variant(source) for _ in range(population_size)]
        evaluation = self.evaluate_variants(method_name, source, population)

        champion_metrics = evaluation["objectives"][evaluation["champion_index"]]
        if champion_metrics.get("correctness", 0.0) < 0.5:
            return None

        champion_code = evaluation["champion_code"]
        if self.hotswap_method(self.owner, method_name, champion_code):
            self.owner.code_version += 1
            return {
                "method": method_name,
                "objectives": champion_metrics,
                "code": champion_code,
            }
        return None