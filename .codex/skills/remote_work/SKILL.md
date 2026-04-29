---
name: remote-ssh-docker-workflow
description: Run remote compile, test, profiling, and debug tasks through an SSH login node and compute node while keeping code edits local and synced through .vscode/sftp.json or sft.json-style SFTP configuration. Use when Codex must validate Hygon/ROCm/DTK environment readiness, check GPU status, inspect Python packages, or execute project commands on the remote compute node. Current project workflow has no Docker layer.
---

# Remote SSH Workflow

This skill keeps code edits local, syncs them to the remote shared filesystem, and executes validation on a compute node through a login node. The current project workflow does **not** use Docker. There is no container path mapping.

## Hard Rules

- Edit code only in the local repository.
- Read connection and remote path settings from the current SFTP config first.
- Do not edit `.vscode/sftp.json`, `sft.json`, shell rc files, modulefiles, conda env files, or any other environment-related files.
- Do not install, remove, or modify remote modules, conda environments, drivers, DTK, RCCL, MPI, or system packages unless the user explicitly asks.
- Sync local changes to `REMOTE_PATH`, then execute on the compute node.
- Do not compile, profile, or run GPU tests on the login node.
- Do not use Docker commands for this workflow.
- Put temporary validation artifacts under the project workspace, preferably `hygon_tmp/`.

## Parameters

Derive these from `.vscode/sftp.json` first. If the project uses a file named `sft.json`, parse the same fields from that file instead. Treat either file as read-only.

- `LOGIN_HOST`: SFTP config `host`
- `LOGIN_USER`: SFTP config `username`
- `LOGIN_PORT`: SFTP config `port`
- `LOGIN_KEY`: SFTP config `privateKeyPath`; copy it to a temporary ACL-restricted file if Windows OpenSSH rejects the original permissions
- `SSH_KEY_ARG`: `-i <key>` for the current workflow
- `LOGIN_TARGET`: `$LOGIN_USER@$LOGIN_HOST`
- `REMOTE_PATH`: SFTP config `remotePath`
- `COMPUTE_HOST`: `gc02r3n15`
- `REMOTE_REPO`: same as `REMOTE_PATH`

Current project values from `.vscode/sftp.json`:

- `LOGIN_HOST` = `10.102.13.101`
- `LOGIN_PORT` = `22`
- `LOGIN_USER` = `zptest`
- `LOGIN_KEY` = `C:/Users/Administrator/.ssh/id_rsa` through a temporary ACL-restricted copy when needed
- `COMPUTE_HOST` = `gc02r3n15`
- `REMOTE_PATH` = `/share_zhipu/home/zptest/yuguo/cuda-optimized-skill`
- `REMOTE_REPO` = `/share_zhipu/home/zptest/yuguo/cuda-optimized-skill`

## Read Parameters Locally

Use PowerShell:

```powershell
$SftpPath = if (Test-Path ".vscode/sftp.json") { ".vscode/sftp.json" } elseif (Test-Path "sft.json") { "sft.json" } else { throw "No SFTP config found" }
$Sftp = Get-Content $SftpPath -Raw | ConvertFrom-Json
$LOGIN_HOST = $Sftp.host
$LOGIN_USER = $Sftp.username
$LOGIN_PORT = $Sftp.port
$LOGIN_KEY_SRC = $Sftp.privateKeyPath
$LOGIN_KEY = Join-Path $env:TEMP "cuda_optimized_skill_id_rsa"
Copy-Item -LiteralPath $LOGIN_KEY_SRC -Destination $LOGIN_KEY -Force
icacls $LOGIN_KEY /inheritance:r /grant:r "$((whoami)):F"
$LOGIN_TARGET = "$LOGIN_USER@$LOGIN_HOST"
$SSH_KEY_ARG = "-i $LOGIN_KEY"
$COMPUTE_HOST = "gc02r3n15"
$REMOTE_PATH = $Sftp.remotePath
$REMOTE_REPO = $REMOTE_PATH
```

Use `ssh -F NUL` and `scp -F NUL` on Windows to ignore local OpenSSH config surprises. Use the temporary ACL-restricted key copy rather than editing the original key file.

## Remote Environment Bootstrap

Every compute-node command that compiles, profiles, tests, or imports project packages must run after this environment setup:

```bash
module unuse /public/software/modules
module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
module load app/rccl/shca_rdma_plugins/v8
module load app/rccl/tests
module load app/miniconda3/25.11.0
conda activate megatron_fla042_mhc_tilelang
```

Use a login shell on the compute node so `module` and `conda` are available. If `conda activate` fails because shell hooks are missing, inspect the existing conda initialization only; do not edit shell startup files. Prefer sourcing the already-installed conda profile script for the command session only.

Do not use `set -u` around the module/conda bootstrap. The observed conda activation scripts can reference unset variables such as `ADDR2LINE`; use `set -eo pipefail` for uploaded validation scripts unless a narrower command has been checked.

## Command Shape

Use this two-hop shape for quick commands:

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'hostname && whoami && pwd'"
```

For project commands, execute a compute-node `bash -lc` block:

```powershell
$RemoteCmd = @'
set -e
cd "$REMOTE_REPO"
module unuse /public/software/modules
module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
module load app/rccl/shca_rdma_plugins/v8
module load app/rccl/tests
module load app/miniconda3/25.11.0
conda activate megatron_fla042_mhc_tilelang
hostname
which hipcc || true
python --version
'@
$Escaped = $RemoteCmd.Replace("'", "'\''")
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'REMOTE_REPO=$REMOTE_REPO bash -lc ''$Escaped'''"
```

If quoting becomes fragile, write the command to a temporary script under `hygon_tmp/`, upload it to `REMOTE_PATH/hygon_tmp/`, then run it on the compute node with `bash`.

Avoid inline multi-hop commands when the remote command contains nested quotes, command substitution such as `$(hostname)`, redirects, here-docs, `printf` format strings, JSON, or shell variables that must expand on the compute node. A direct local PowerShell -> login shell -> compute shell command can lose quote boundaries and fail with messages like `unexpected EOF while looking for matching '"'`. In that case, create a script locally under `hygon_tmp/`, upload it to `REMOTE_PATH/hygon_tmp/`, and execute that script on `gc02r3n15`.

## Quick Verification Workflow

1. Verify login-node SSH and shared path.

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "hostname && whoami && test -d $REMOTE_PATH && echo remote_path_ok=$REMOTE_PATH"
```

2. Verify compute-node SSH from the login node and shared storage visibility.

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'hostname && whoami && test -d $REMOTE_PATH && echo shared_remote_path_ok=$REMOTE_PATH'"
```

3. Verify module and conda environment on the compute node.

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash -lc `"cd $REMOTE_PATH && module unuse /public/software/modules && module load compiler/dtk/25.04.4 && module load mpi/hpcx/2.18.0/gcc-8.5.0/shca && module load app/rccl/shca_rdma_plugins/v8 && module load app/rccl/tests && module load app/miniconda3/25.11.0 && conda activate megatron_fla042_mhc_tilelang && hostname && which hipcc && hipcc --version | head -n 2 && python --version`"'"
```

4. Check GPU/DTK status on the compute node after loading the environment.

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash -lc `"cd $REMOTE_PATH && module unuse /public/software/modules && module load compiler/dtk/25.04.4 && module load app/miniconda3/25.11.0 && conda activate megatron_fla042_mhc_tilelang && (rocm-smi --showuse --showmemuse || rocm-smi || true) && (rocminfo | grep -m1 -E \"gfx[0-9]+\" || true)`"'"
```

Choose a device with low memory and compute use, then set `HIP_VISIBLE_DEVICES=<device>` for compile/run/test commands that need an isolated GPU.

## Compile, Test, Profile Templates

Compile check:

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash -lc `"cd $REMOTE_PATH && module unuse /public/software/modules && module load compiler/dtk/25.04.4 && module load app/miniconda3/25.11.0 && conda activate megatron_fla042_mhc_tilelang && python -m compileall .`"'"
```

Targeted Python test:

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash -lc `"cd $REMOTE_PATH && module unuse /public/software/modules && module load compiler/dtk/25.04.4 && module load app/miniconda3/25.11.0 && conda activate megatron_fla042_mhc_tilelang && HIP_VISIBLE_DEVICES=<device> PYTHONPATH=. pytest -q <path/to/test.py>`"'"
```

HIP compile and run:

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash -lc `"cd $REMOTE_PATH && module unuse /public/software/modules && module load compiler/dtk/25.04.4 && module load app/miniconda3/25.11.0 && conda activate megatron_fla042_mhc_tilelang && hipcc -O2 hip_vector_add.cpp -o hygon_tmp/hip_vector_add && HIP_VISIBLE_DEVICES=<device> hygon_tmp/hip_vector_add`"'"
```

Hygon profiler command:

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash -lc `"cd $REMOTE_PATH && module unuse /public/software/modules && module load compiler/dtk/25.04.4 && module load app/miniconda3/25.11.0 && conda activate megatron_fla042_mhc_tilelang && HIP_VISIBLE_DEVICES=<device> python skills/hyhon-hip-kernel-optimizer/scripts/profile_hipprof.py --state <run_dir>/state.json --iter 1 --which kernel --pmc-mode all`"'"
```

## Sync Guidance

- Keep the SFTP config read-only. Do not edit it to fix workflow issues.
- Save local files, let `uploadOnSave` sync, then verify the file remotely.
- If auto-sync is delayed, upload explicit files with `scp` to `REMOTE_PATH`.

```powershell
scp -F NUL -P $LOGIN_PORT $SSH_KEY_ARG <local_file> "${LOGIN_TARGET}:$REMOTE_PATH/<relative_target>"
```

Verify on login and compute nodes:

```powershell
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ls -l $REMOTE_PATH/<relative_target>"
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'ls -l $REMOTE_PATH/<relative_target>'"
```

For generated remote artifacts that the user needs locally, copy them back with `scp` from the login node path:

```powershell
scp -F NUL -P $LOGIN_PORT $SSH_KEY_ARG -r "${LOGIN_TARGET}:$REMOTE_PATH/hygon_tmp/<artifact>" "hygon_tmp/"
```

For remote artifact generation plus pullback, prefer this script-based pattern:

```powershell
# 1. Write a small local script under hygon_tmp/.
# 2. Upload it to $REMOTE_PATH/hygon_tmp/.
# 3. Execute it through the login node on gc02r3n15.
# 4. Pull the generated $REMOTE_PATH/hygon_tmp/<artifact> directory back to local hygon_tmp/.
scp -F NUL -P $LOGIN_PORT $SSH_KEY_ARG hygon_tmp/create_artifact.sh "${LOGIN_TARGET}:$REMOTE_PATH/hygon_tmp/create_artifact.sh"
ssh -F NUL -p $LOGIN_PORT $SSH_KEY_ARG $LOGIN_TARGET "ssh $COMPUTE_HOST 'bash $REMOTE_PATH/hygon_tmp/create_artifact.sh <artifact-name>'"
scp -F NUL -P $LOGIN_PORT $SSH_KEY_ARG -r "${LOGIN_TARGET}:$REMOTE_PATH/hygon_tmp/<artifact-name>" "hygon_tmp/"
```

## Windows SSH Reliability Notes

- Prefer `ssh -F NUL ...` and `scp -F NUL ...` to bypass local `~/.ssh/config`.
- If OpenSSH reports bad permissions on the private key, create a temporary copy with restricted ACLs and use that copy for this session.

```powershell
$LOGIN_KEY_SRC = $Sftp.privateKeyPath
$LOGIN_KEY = Join-Path $env:TEMP "cuda_optimized_skill_id_rsa"
Copy-Item -LiteralPath $LOGIN_KEY_SRC -Destination $LOGIN_KEY -Force
icacls $LOGIN_KEY /inheritance:r /grant:r "$((whoami)):F"
```

## Failure Handling

- If login-node SSH fails, check local SSH key/config/ACL issues first; do not edit remote environment files.
- If login-node to compute-node SSH fails, verify `ssh gc02r3n15 hostname` from the login node. Do not assume direct local access to `gc02r3n15`.
- If `REMOTE_PATH` exists on the login node but not on `gc02r3n15`, stop and report a shared-storage issue.
- If `module` is unavailable, use a compute-node login shell and inspect existing shell initialization. Do not modify shell rc files or module configuration.
- If `conda activate megatron_fla042_mhc_tilelang` fails, report the error and inspect existing conda/module state. Do not create, update, or remove conda environments.
- If `hipcc`, `rocminfo`, `rocm-smi`, `hipprof`, or `dccobjdump` is unavailable after the documented module loads, treat it as an environment mismatch and report it instead of changing modules or installing packages.
- If package or import checks fail, separate environment mismatch from code regression and ask before changing any environment-related file.
- If a multi-hop inline command fails with quote-related shell errors such as `unexpected EOF`, `syntax error: unexpected end of file`, or missing remote artifact paths after a failed creation command, switch to the `hygon_tmp` script upload pattern. Do not keep adding escaping layers blindly.
