"""Load one requested smoke package without installed-package name resolution."""
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import uuid


def candidate_root(path):
    """The smoke entrypoint requires a regular package exporting its operators."""
    root = Path(path).resolve()
    if not root.is_dir() or not (root / '__init__.py').is_file():
        raise ValueError('Candidate root must be a directory containing __init__.py: ' + str(root))
    if (root / '__init__.py').resolve().parent != root:
        raise ValueError('Candidate package initializer must be inside the requested root')
    return root


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


def source_binding(requested, imported, requested_sha256=None):
    requested = Path(requested).resolve(strict=True)
    imported = Path(imported).resolve(strict=True)
    if requested != imported:
        raise ValueError(f'Imported source differs from requested source: {imported} != {requested}')
    digest = hashlib.sha256(imported.read_bytes()).hexdigest()
    if requested_sha256 is not None and requested_sha256 != digest:
        raise ValueError('Candidate source changed during import: ' + str(imported))
    return {'requested_path': str(requested), 'imported_path': str(imported),
            'requested_sha256': requested_sha256 or digest, 'imported_sha256': digest}


def verify_sources_unchanged(binding):
    current = source_hashes(Path(binding['resolved_root']))
    binding['sources_unchanged'] = current == binding['requested_sources']
    if not binding['sources_unchanged']:
        binding['current_sources'] = current
        raise ValueError('Candidate Python sources changed during the smoke run')


@contextmanager
def load_candidate_package(path, operators):
    """Keep relative imports isolated for the lifetime of the smoke run."""
    root = candidate_root(path)
    requested_sources = source_hashes(root)
    name = '_camodel_candidate_' + uuid.uuid4().hex
    cache = tempfile.TemporaryDirectory(prefix='camodel-source-import-')
    saved_cache_prefix = sys.pycache_prefix
    try:
        # A timestamp/size-valid old .pyc can otherwise execute different bytes.
        # Keep a fresh cache through late relative imports, without deleting or
        # writing caches in the requested package. The CLI owns this process.
        sys.pycache_prefix = cache.name
        spec = importlib.util.spec_from_file_location(
            name, root / '__init__.py', submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        binding = {'requested_root': str(Path(path).absolute()), 'resolved_root': str(root),
                   'requested_sources': requested_sources,
                   'package': source_binding(root / '__init__.py', package.__spec__.origin,
                                             requested_sources['__init__.py']),
                   'operators': {}}
        for op in operators:
            fn = getattr(package, op, None)
            module_name = getattr(fn, '__module__', '')
            if not inspect.isfunction(fn) or not (
                module_name == name or module_name.startswith(name + '.')
            ):
                raise ValueError(f'{op}: expected a Python wrapper from the requested package')
            module = sys.modules[module_name]
            relative = module_name[len(name):].lstrip('.').split('.') if module_name != name else []
            requested = root.joinpath(*relative)
            requested = (requested / '__init__.py' if module.__spec__.submodule_search_locations is not None
                         else requested.with_suffix('.py'))
            origin = Path(module.__spec__.origin).resolve(strict=True)
            if not origin.is_relative_to(root):
                raise ValueError(f'{op}: imported source is outside the requested root: {origin}')
            requested_sha = requested_sources.get(str(requested.relative_to(root)))
            if requested_sha is None:
                raise ValueError(f'{op}: source was not present before import')
            record = source_binding(requested, origin, requested_sha)
            source_binding(origin, fn.__code__.co_filename)
            binding['operators'][op] = {**record, 'module': module_name,
                                       'function_code_path': str(Path(fn.__code__.co_filename).resolve())}
        yield package, binding
    finally:
        sys.pycache_prefix = saved_cache_prefix
        for loaded in list(sys.modules):
            if loaded == name or loaded.startswith(name + '.'):
                del sys.modules[loaded]
        cache.cleanup()
