# Hygon DCU Kernel Optimizer Plugin

This Codex plugin packages the Hygon DCU kernel workflow as reusable skills:

- `hygon-hip-baseline-generator`: generate a correctness-first HIP baseline from a Torch, Triton, TileLang, or Python reference plus shape JSON.
- `hygon-hip-kernel-optimizer`: iteratively optimize HIP/CK Tile kernels with DTK/HIP tooling, `hipprof`, PMC/SQTT analysis, and `dccobjdump` ISA checks.

The plugin does not include a remote execution skill. Each target project must provide its own remote workflow because login nodes, compute nodes, Docker usage, module setup, schedulers, and sync rules vary between projects.

## User-Level Install

Manual Codex plugin loading uses the installed plugin cache shape. Copy the plugin to:

```text
%USERPROFILE%\.codex\plugins\cache\codex-local-plugins\hygon-dcu-kernel-optimizer\local
```

PowerShell:

```powershell
$src = "D:\git\cuda-optimized-skill\plugins\hygon-dcu-kernel-optimizer"
$dst = "$HOME\.codex\plugins\cache\codex-local-plugins\hygon-dcu-kernel-optimizer\local"
New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
```

Optionally keep a source copy at:

```text
%USERPROFILE%\.codex\plugins\hygon-dcu-kernel-optimizer
```

but the cache path above is the one that matches already-installed plugins such as GitHub and Superpowers.

Create or update:

```text
%USERPROFILE%\.codex\plugins\marketplace.json
```

with:

```json
{
  "name": "codex-local-plugins",
  "interface": {
    "displayName": "Codex Local Plugins"
  },
  "plugins": [
    {
      "name": "hygon-dcu-kernel-optimizer",
      "source": {
        "source": "local",
        "path": "./hygon-dcu-kernel-optimizer"
      },
      "policy": {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL"
      },
      "category": "Developer Tools"
    }
  ]
}
```

Restart or refresh Codex after installation.

Enable the plugin in:

```text
%USERPROFILE%\.codex\config.toml
```

Add:

```toml
[plugins."hygon-dcu-kernel-optimizer@codex-local-plugins"]
enabled = true
```

Codex loads the plugin and skill list when a session starts, so restart or refresh Codex after changing `marketplace.json` or `config.toml`.

After restart, the enabled plugin id is:

```text
hygon-dcu-kernel-optimizer@codex-local-plugins
```

The plugin contributes these skills:

```text
hygon-dcu-kernel-optimizer:hygon-hip-baseline-generator
hygon-dcu-kernel-optimizer:hygon-hip-kernel-optimizer
```

## Project-Level Install

For a single target project, copy the plugin into that project:

```powershell
Copy-Item -Recurse D:\git\cuda-optimized-skill\plugins\hygon-dcu-kernel-optimizer <target-project>\plugins\
```

Then create or update:

```text
<target-project>\.agents\plugins\marketplace.json
```

with the same marketplace entry, keeping:

```json
"path": "./plugins/hygon-dcu-kernel-optimizer"
```

For repo-local marketplaces, enable the plugin in the Codex config using the marketplace name from that project's `.agents/plugins/marketplace.json`. For example, if the marketplace name is `local-plugins`:

```toml
[plugins."hygon-dcu-kernel-optimizer@local-plugins"]
enabled = true
```

Restart or refresh Codex after enabling it.

## Using The Plugin

In a target project, invoke the plugin from Codex with a prompt like:

```text
@Hygon DCU Kernel Optimizer
Use the Hygon baseline generator with ref.py and shape {"N":1048576}.
Validate correctness using this project's remote workflow, then optimize for 3 iterations.
```

When the plugin is used from another repository, command examples inside the skills should be resolved from the loaded plugin path, not from `<target-project>\skills\...`. In particular, `inspect_ref.py`, `generate_baseline.py`, `preflight.py`, `benchmark.py`, and `orchestrate.py` live under the plugin's `skills\...` directories.

For an existing HIP kernel:

```text
@Hygon DCU Kernel Optimizer
Optimize kernels/my_kernel.hip against refs/ref.py with dims {"M":1024,"N":1024,"K":1024}.
Run 3 iterations and use the target project's remote workflow for DCU validation.
```

## Target Project Scratch Directory

When used in another project, the plugin expects a scratch directory at the target project root:

```text
hygon_tmp/
```

The plugin includes:

```text
scripts/ensure_hygon_workspace.py
```

Run it from the target project when needed:

```powershell
python <plugin-root>\scripts\ensure_hygon_workspace.py --root <target-project>
```

It creates `hygon_tmp/` and adds `hygon_tmp/` to the target project's `.gitignore` unless `--no-gitignore` is passed.

## Remote Workflow Rule

Before running remote validation, Codex should read the target project's own instructions, such as:

- `AGENTS.md`
- `.codex/skills/`
- `.agents/skills/`
- README or runbook files
- scheduler/module/Docker notes

Do not copy the remote workflow from this plugin source repository into another project unless that project explicitly uses the same environment.

## Updating The Installed Plugin

After editing the source plugin, reinstall by replacing the user-level copy:

```powershell
$src = "D:\git\cuda-optimized-skill\plugins\hygon-dcu-kernel-optimizer"
$dst = "$HOME\.codex\plugins\cache\codex-local-plugins\hygon-dcu-kernel-optimizer\local"
if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
```

Restart or refresh Codex after updating.

If the plugin does not appear in the `@` menu, check:

1. The plugin directory exists under `%USERPROFILE%\.codex\plugins\hygon-dcu-kernel-optimizer`.
2. The installed cache directory exists under `%USERPROFILE%\.codex\plugins\cache\codex-local-plugins\hygon-dcu-kernel-optimizer\local`.
3. `%USERPROFILE%\.codex\plugins\marketplace.json` contains the plugin entry.
4. `%USERPROFILE%\.codex\config.toml` contains:

```toml
[plugins."hygon-dcu-kernel-optimizer@codex-local-plugins"]
enabled = true
```

5. Codex was restarted after the config change.
