"""Versioned operator contracts for the shared CI lifecycle (no network)."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def load_recipe(name, directory=None):
    directory = directory or HERE/'operators'
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError('Invalid operator recipe name')
    path = directory/(name+'.json')
    if not path.is_file():
        raise ValueError('Unsupported operator recipe: '+name)
    recipe = json.loads(path.read_text())
    if (recipe.get('function') != name or not isinstance(recipe.get('operator'), str)
            or not recipe['operator'] or type(recipe.get('numerical_cases')) is not int
            or recipe['numerical_cases'] < 1):
        raise ValueError('Invalid operator recipe contract')
    for key in ('prompt', 'numerical_probe'):
        resource = recipe.get(key)
        if not isinstance(resource, str) or Path(resource).name != resource or not (directory/resource).is_file():
            raise ValueError('Missing or unsafe operator resource: '+key)
    reference = recipe.get('reference')
    if reference is not None and (not isinstance(reference, str) or Path(reference).is_absolute() or '..' in Path(reference).parts):
        raise ValueError('Unsafe reference candidate path')
    return recipe


def run_recipe(root):
    """Resume uses the generation snapshot, never the dispatch selection."""
    path = root/'run.json'
    meta = json.loads(path.read_text()) if path.is_file() else {}
    name = meta.get('operator_recipe')
    if name is None:  # Runs predating recipes were strictly SwiGlu.
        return {'operator':'SwiGlu', 'function':'swi_glu', 'numerical_cases':12}
    return load_recipe(name, root/'source/integrations/cannbench/ci/operators')
