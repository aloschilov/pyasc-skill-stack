# Installing opencode on the remote NPU box (Turing Yellow Zone)

This mirrors the local "cloud configuration" (see
[`docker/opencode-profiles/README.md`](../docker/opencode-profiles/README.md),
profile `cloud-default.json`) but targets the **remote NPU container** reached
through `gcpty`, wired to the internal **Turing Yellow Zone** LLM gateway instead
of Alibaba DashScope.

- Local machine → `opencode` (bun global) → DashScope `dashscope/glm-5`
- Remote gcpty box → `opencode` (standalone binary) → Yellow Zone `yellowzone/GLM-5.2`

All remote commands below are run through the `gcpty` relay, e.g.:

```bash
cd ~/workspace/gitcode-pty
python -m gcpty client https://gitcode.com/compiler-team/gitcode-pty/pull/2 -- '<remote command>'
```

## Target machine (as probed)

| Property | Value |
|----------|-------|
| Host | `c50f972f7149` (Ubuntu 22.04, `x86_64`) |
| User | `root` (`HOME=/root`) |
| Pre-installed | `curl`, `unzip`, `apt-get` (no `node`/`bun`/`npm`/`opencode`) |
| HTTPS egress | via Huawei HIS proxy `http_proxy=http://…@172.18.100.92:8080` |
| Yellow Zone gateway | `http://7.150.6.255:3300/v1` (internal; must **bypass** the proxy) |

Two environment quirks drive the setup:

1. **CA bundle was missing** — `/etc/ssl/certs/ca-certificates.crt` did not exist,
   so every HTTPS request failed with `curl: (77) error setting certificate file`.
2. **The internal gateway must skip the proxy** — the corporate `no_proxy` only
   lists the CIDR `7.0.0.0/8`, which `node`/`undici` (the runtime behind
   opencode's `openai-compatible` provider) does **not** honour. The explicit host
   `7.150.6.255` must be added or every request hangs indefinitely through the
   HIS proxy.

## Step 1 — fix the CA bundle (enables HTTPS)

```bash
apt-get install -y --no-install-recommends ca-certificates
# creates /etc/ssl/certs/ca-certificates.crt (~182 KB)
```

This is required so opencode can (a) be downloaded and (b) fetch its
`@ai-sdk/openai-compatible` provider package from the npm registry on first run.

## Step 2 — install the opencode standalone binary

The box has no `bun`/`node`, so instead of the bun-global install used locally we
drop in the prebuilt CLI binary (pinned to the same version as local, `1.18.4`,
from the `anomalyco/opencode` release):

```bash
cd /root
curl -k -sSL -o /root/opencode.tgz \
  https://github.com/anomalyco/opencode/releases/download/v1.18.4/opencode-linux-x64.tar.gz
mkdir -p /usr/local/lib/opencode
tar xzf /root/opencode.tgz -C /usr/local/lib/opencode
# -> /usr/local/lib/opencode/opencode  (single ~178 MB binary)
```

> `-k` (insecure) is only needed if you install the binary *before* Step 1.
> After the CA bundle is in place plain `curl` works.

## Step 3 — install a proxy-bypass launcher on PATH

Rather than a bare symlink, install a tiny wrapper at `/usr/local/bin/opencode`
so the internal gateway always bypasses the HIS proxy (external npm/github egress
is left on the proxy untouched):

```bash
cat > /usr/local/bin/opencode <<'EOF'
#!/bin/bash
# opencode launcher for the Turing Yellow Zone box.
# The internal LLM gateway (7.150.6.255:3300) must bypass the Huawei HIS proxy;
# the corporate no_proxy only lists a CIDR (7.0.0.0/8) which node/undici does not
# honour, so add the explicit host. External egress still uses the proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}7.150.6.255"
export no_proxy="${no_proxy:+$no_proxy,}7.150.6.255"
exec /usr/local/lib/opencode/opencode "$@"
EOF
chmod +x /usr/local/bin/opencode
opencode --version   # -> 1.18.4
```

## Step 4 — configure the Yellow Zone provider

Write the global config `/root/.config/opencode/opencode.json`. This mirrors
`cloud-default.json` (self-contained provider + inline key + baseline
permissions) but points at the Yellow Zone endpoint and pins
`yellowzone/GLM-5.2`. The committed template
[`docker/opencode-profiles/yellowzone-glm-5.2.json`](../docker/opencode-profiles/yellowzone-glm-5.2.json)
uses `${YELLOWZONE_API_KEY}` so no secret is committed; on the remote box the
literal key from the **Turing Yellow Zone credentials card** is inlined:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "yellowzone": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Turing Yellow Zone",
      "options": {
        "baseURL": "http://7.150.6.255:3300/v1",
        "apiKey": "sk-…"           // from the Yellow Zone credentials card
      },
      "models": {
        "GLM-5.2":           { "name": "GLM-5.2" },
        "DeepSeek-V4-Pro":   { "name": "DeepSeek-V4-Pro" },
        "DeepSeek-V4-Flash": { "name": "DeepSeek-V4-Flash" }
      }
    }
  },
  "model": "yellowzone/GLM-5.2",
  "permission": { "read": "allow", "edit": "allow", "bash": "allow",
                  "glob": "allow", "grep": "allow", "list": "allow",
                  "skill": "allow", "task": "allow",
                  "external_directory": { "*": "deny",
                                          "/home/l00958488/pyasc-fork1/*": "allow" } }
}
```

The available model ids were confirmed straight from the gateway (proxy
bypassed):

```bash
curl -sS --noproxy '*' http://7.150.6.255:3300/v1/models \
  -H "Authorization: Bearer sk-…"
# -> DeepSeek-V4-Flash, DeepSeek-V4-Pro, GLM-5.2
```

`GLM-5.2` is the renamed "GLM5" model from the Yellow Zone notice.

## Step 5 — verify

```bash
opencode --version
# 1.18.4

cd /tmp && opencode run "Reply with exactly: HELLO_FROM_YELLOWZONE" \
  --model yellowzone/GLM-5.2
# > build · GLM-5.2
# HELLO_FROM_YELLOWZONE
```

## Gotchas learned during setup

- **Always wrap remote `opencode run` in `timeout N`.** If a request hangs
  (e.g. proxy bypass missing), `opencode run` does **not** self-timeout; the
  process lives indefinitely and, because `gcpty` executes serially, it blocks
  every subsequent command in the session. Recover with `killall -9 opencode`.
- **Do not kill the local `gcpty client` mid-run** to escape a hang — the remote
  process keeps running and wedges the queue. Prefer `timeout` on the remote
  side so the command ends cleanly.
- **First run is slow** (~40–120 s): opencode fetches the
  `@ai-sdk/openai-compatible` provider package (needs Step 1 + the proxy) before
  the first completion.
