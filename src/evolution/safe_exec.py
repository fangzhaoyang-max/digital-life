"""
Safe execution environment for evolved code
"""

import ast
import textwrap
import types
import uuid
import linecache
from typing import Dict, Any, Optional

from ..imports import logger


class ASTSafetyError(Exception):
    pass


class ASTSafetyChecker(ast.NodeVisitor):
    """AST safety checker for evolved code"""
    
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
    """Safe execution environment for evolved code"""
    
    ALLOWED_BUILTINS = {
        'len': len, 'min': min, 'max': max, 'sum': sum, 'range': range,
        'enumerate': enumerate, 'any': any, 'all': all, 'abs': abs,
        'float': float, 'int': int, 'str': str, 'bool': bool,
        'sorted': sorted, 'zip': zip, 'map': map, 'filter': filter, 'reversed': reversed,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'print': print,
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'TimeoutError': TimeoutError,
        'getattr': getattr, 'setattr': setattr, 'hasattr': hasattr, 'callable': callable,
    }

    @staticmethod
    def _cleanup_linecache():
        """Limit mutation source code in linecache"""
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

        # Safe globals with necessary dependencies
        g = {
            "__builtins__": SafeExec.ALLOWED_BUILTINS,
            "time": __import__('time'),
            "random": __import__('random'),
            "logger": logger,
            "copy": __import__('copy'),
            "inspect": __import__('inspect'),
            "__quantum_var__": __import__('random').randint(0, 1),
            "quantum_flag": __import__('random').randint(0, 1),
            "break_flag": False,
        }
        
        # Add numpy and ML components if available
        try:
            import numpy as np
            g["np"] = np
        except ImportError:
            pass
            
        try:
            from ..imports import KMeans, StandardScaler
            g["KMeans"] = KMeans
            g["StandardScaler"] = StandardScaler
        except ImportError:
            pass
            
        # Add LifeState enum
        try:
            from ..core.life_states import LifeState
            g["LifeState"] = LifeState
        except ImportError:
            pass

        if extra_globals:
            g.update(extra_globals)

        # Add to linecache for inspect.getsource
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