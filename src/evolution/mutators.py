"""
Mutation utilities for the Digital Life system
"""

import ast
import random

from ..imports import fx, nn, torch


class NeuralASTTransformer(ast.NodeTransformer):
    """AST transformer guided by neural hotspots"""

    def __init__(self, hotspots):
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
        if len(node.targets) == 1 and isinstance(node.value, ast.Constant):
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


class FXGraphMutator:
    """Mutator for torch.fx graphs"""
    ACTS = []
    if nn is not None:
        ACTS = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU]

    @staticmethod
    def mutate(graph_module):
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
                            mod_path = n.target
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
