# Hygon DCU Kernel Optimizer Plugin

This Codex plugin packages the Hygon DCU kernel workflow as reusable skills:

- `hygon-hip-baseline-generator`: generate a correctness-first HIP baseline from a Torch, Triton, TileLang, or Python reference plus shape JSON.
- `hyhon-hip-kernel-optimizer`: iteratively optimize HIP/CK Tile kernels with DTK/HIP tooling, `hipprof`, PMC/SQTT analysis, and `dccobjdump` ISA checks.

The plugin does not include a remote execution skill. Each target project must provide its own remote workflow because login nodes, compute nodes, Docker usage, module setup, schedulers, and sync rules vary between projects.

## User-Level Install

Copy the plugin to your Codex user plugin directory:

```powershell
$src = "D:\git\cuda-optimized-skill\plugins\hygon-dcu-kernel-optimizer"
$dst = "$HOME\.codex\plugins\hygon-dcu-kernel-optimizer"
New-Item -ItemType Directory -Force "$HOME\.codex\plugins" | Out-Null
if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
```

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

## Using The Plugin

In a target project, invoke the plugin from Codex with a prompt like:

```text
@Hygon DCU Kernel Optimizer
Use the Hygon baseline generator with ref.py and shape {"N":1048576}.
Validate correctness using this project's remote workflow, then optimize for 3 iterations.
```

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
$dst = "$HOME\.codex\plugins\hygon-dcu-kernel-optimizer"
if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
```

Restart or refresh Codex after updating.
