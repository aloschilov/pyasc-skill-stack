"""Explicit profile identity and credential binding for the shared executor.

No network discovery or model fallback. Resolve only the endpoint/credential
references selected by the hashed profile, never unrelated ambient variables.
"""
from copy import deepcopy
import ast
import json
import re
from urllib.parse import urlsplit


def require(condition, message):
    if not condition: raise ValueError(message)


def variable(value):
    match = re.fullmatch(r'(?:\{env:([A-Z_][A-Z0-9_]*)\}|\$\{([A-Z_][A-Z0-9_]*)\})', value)
    return next((v for v in match.groups() if v), None) if match else None


def endpoint(value):
    require(isinstance(value, str) and not any(c.isspace() for c in value), 'Invalid selected provider endpoint')
    try:
        url = urlsplit(value)
        port = url.port  # urlsplit alone does not validate malformed/out-of-range ports.
    except ValueError: raise ValueError('Invalid selected provider endpoint') from None
    require(url.scheme in ('http', 'https') and url.hostname and not url.username and not url.password
        and not url.query and not url.fragment and (port is None or port > 0), 'Invalid selected provider endpoint')
    return value.rstrip('/')


def configuration(config):
    """Public, versioned declaration; environment values are never read here."""
    require(isinstance(config, dict), 'Invalid provider profile')
    model = config.get('model')
    require(isinstance(model, str) and re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+', model),
        'Profile must select an explicit provider/model')
    provider, name = model.split('/')
    providers = config.get('provider', {})
    require(isinstance(providers, dict) and set(providers) == {provider}, 'Bounded profiles must declare only the selected provider')
    selected = providers[provider]
    require(isinstance(selected, dict) and selected.get('npm') == '@ai-sdk/openai-compatible', 'Unsupported provider transport; no fallback')
    require(isinstance(selected.get('models'), dict) and name in selected['models'], 'Selected model is absent from profile')
    options = selected.get('options', {})
    require(isinstance(options, dict) and set(options) == {'baseURL', 'apiKey'} and all(isinstance(v, str) for v in options.values()),
        'Explicit endpoint and credential references required; custom provider options are unsupported')
    url, key = options['baseURL'], options['apiKey']
    url_var, key_var = variable(url), variable(key)
    require(key_var is not None or provider == 'ollama' and key == 'ollama',
        'Credential must be an environment reference; literal secrets are forbidden')
    entry = selected['models'][name]
    require(isinstance(entry, dict) and isinstance(entry.get('options', {}), dict)
        and isinstance(entry.get('limit', {}), dict), 'Invalid selected model settings')
    return {'schema_version': 1, 'provider': provider, 'transport': selected['npm'],
        'endpoint': {'environment': url_var} if url_var else {'literal': endpoint(url)},
        'credential': {'environment': key_var} if key_var else {'public_placeholder': 'ollama'},
        'model_options': deepcopy(entry.get('options', {})), 'model_limits': deepcopy(entry.get('limit', {})),
        'provider_revision': None,
        'revision_note': 'Response model identity is checked separately; an immutable provider revision is not asserted.'}


def resolve(config, planned_model, environment):
    """Return secrets only to the gate caller; errors contain names, not values."""
    declared = configuration(config)
    require(config['model'] == planned_model['requested'] and declared == planned_model['configuration'],
        'Selected profile/model configuration differs from plan')
    def value(binding):
        if 'environment' in binding:
            name = binding['environment']; result = environment.get(name)
            require(isinstance(result, str) and bool(result.strip()), 'Missing selected provider variable '+name)
            return result
        return binding.get('literal', binding.get('public_placeholder'))
    key = value(declared['credential'])
    require(not any(c in key for c in ('\r', '\n')), 'Invalid selected provider credential')
    return endpoint(value(declared['endpoint'])), key


def inspect_binding(model, sources, read):
    """Read-only validation, including legacy receipts without binding proof.

    New executor source closures include this helper and must include both
    fields. Legacy archives remain inspectable; this never resolves credentials
    or authorizes execution of their plans.
    """
    from generation_plan import digest
    fields = {'profile_sha256', 'configuration'}
    present = fields.intersection(model)
    # The hashed planner independently declares the binding dependency. Removing
    # just the helper and metadata cannot disguise a new receipt as a legacy one.
    planner = ast.parse(read('tests/tools/generation_plan.py'))
    uses_binding = any(isinstance(node, ast.ImportFrom) and node.module == 'generation_models' for node in ast.walk(planner))
    require(uses_binding == ('tests/tools/generation_models.py' in sources), 'Incomplete model binding source closure')
    if not uses_binding:
        require(not present, 'Model binding lacks its source implementation')
        return None
    require(present == fields, 'Missing model profile binding')
    path = 'integrations/opencode/profiles/'+model['profile']+'.json'
    raw = read(path)
    require(model['profile_sha256'] == sources.get(path) == digest(raw), 'Model profile binding hash mismatch')
    config = json.loads(raw)
    require(config['model'] == model['requested'] and configuration(config) == model['configuration'],
        'Model profile binding configuration mismatch')
    return 1


def records_wire_settings(provider_source):
    """The hashed provider implementation declares whether legacy receipts need settings."""
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'request_settings'
        for node in ast.walk(ast.parse(provider_source)))


def inspect_request_settings(request, cap, required):
    """Validate the public allowlist; never infer settings for historical requests."""
    from generation_provider import request_settings
    if not required:
        require('request_settings' not in request, 'Wire settings lack their source implementation')
        return None
    value = request.get('request_settings')
    require(isinstance(value, dict), 'Missing recorded wire settings')
    require(type(value.get('schema_version')) is int and value['schema_version'] == 1, 'Unknown wire settings schema')
    fields = value.get('present_fields')
    require(isinstance(fields, list) and all(isinstance(n, str) for n in fields)
        and fields == sorted(set(fields)), 'Invalid recorded setting fields')
    sampling, reasoning = value.get('sampling'), value.get('reasoning')
    require(isinstance(sampling, dict) and isinstance(reasoning, dict), 'Invalid recorded settings')
    values = dict(sampling, **reasoning)
    require(set(fields) <= set(values), 'Unknown recorded setting field')
    body = {n: values[n] for n in fields}
    body['max_tokens'] = cap
    require(type(cap) is int and cap > 0 and type(value.get('max_tokens')) is int and type(request.get('max_tokens')) is int
        and request['max_tokens'] == cap and value == request_settings(body), 'Recorded wire settings/cap mismatch')
    return value
