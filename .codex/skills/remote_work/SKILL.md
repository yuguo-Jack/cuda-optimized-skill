---
name: remote-ssh-docker-workflow
description: Run remote compile, test, profiling, and debug tasks through SSH plus docker exec while keeping code edits local and synced to the remote node. Use when Codex must validate environment readiness, check ROCm/DTK/Hygon GPU card status, inspect Python packages, verify host-to-container workspace mounts, or execute project commands inside a remote container. Defaults for this cuda-optimized-skill workspace are hg@10.17.176.13:22, Docker container megamoe, host /home/hg/yuguo mapped to container /workspace, and repo path /home/hg/yuguo/cuda-optimized-skill mapped to /workspace/cuda-optimized-skill.
---

# Remote SSH Docker Workflow (Simplified)

## Execution Contract

Follow this rule for all downstream remote skills:

- Edit code only in the local repository.
- Sync local changes to remote using `.vscode/sftp.json` (`uploadOnSave: true`) or explicit upload commands.
- Execute remote work only with `ssh ... "docker exec ... bash -lc 'source /opt/dtk/env.sh && <cmd>'"`.
- Every `docker exec` command that runs inside the container must source DTK first with `source /opt/dtk/env.sh && ...`.
- Host-side Docker management commands such as `docker ps`, `docker inspect`, and `docker start` do not run inside the container and do not source `/opt/dtk/env.sh`.
- Avoid direct remote-host compilation and testing outside Docker unless explicitly requested.
- For this workspace, the host mount root is `/home/hg/yuguo` and the container mount root is `/workspace`.
- The local project may not be uploaded yet. Before first upload, remote `/home/hg/yuguo/cuda-optimized-skill` and container `/workspace/cuda-optimized-skill` may be missing; treat that as setup state, not an environment failure.

## Parameters

Derive connection details from `.vscode/sftp.json` first instead of hardcoding host/user/key:

- `SSH_HOST`: read from `.vscode/sftp.json` field `host`
- `SSH_USER`: read from `.vscode/sftp.json` field `username`
- `SSH_PORT`: read from `.vscode/sftp.json` field `port`
- `SSH_KEY`: read from `.vscode/sftp.json` field `privateKeyPath`
- `SSH_TARGET`: `$SSH_USER@$SSH_HOST`
- `DOCKER_NAME`: default `megamoe` unless the project specifies otherwise
- `REMOTE_PATH`: read from `.vscode/sftp.json` field `remotePath`
- `HOST_MOUNT_ROOT`: default `/home/hg/yuguo`
- `CONTAINER_WORKSPACE`: default `/workspace`
- `CONTAINER_REPO`: replace the `HOST_MOUNT_ROOT` prefix in `REMOTE_PATH` with `CONTAINER_WORKSPACE`; for this project that yields `/workspace/cuda-optimized-skill`

For this project:

- `host` = `10.17.176.13`
- `username` = `hg`
- `port` = `22`
- `privateKeyPath` = `C:/Users/Administrator/.ssh/id_rsa`
- `remotePath` = `/home/hg/yuguo/cuda-optimized-skill`
- `DOCKER_NAME` = `megamoe`
- `HOST_MOUNT_ROOT` = `/home/hg/yuguo`
- `CONTAINER_WORKSPACE` = `/workspace`
- `CONTAINER_REPO` = `/workspace/cuda-optimized-skill`

## Read Parameters From sftp.json

Read and derive paths from local PowerShell:

```powershell
$Sftp = Get-Content .vscode/sftp.json -Raw | ConvertFrom-Json
$SSH_HOST = $Sftp.host
$SSH_USER = $Sftp.username
$SSH_PORT = $Sftp.port
$SSH_KEY = $Sftp.privateKeyPath
$SSH_TARGET = "$SSH_USER@$SSH_HOST"
$DOCKER_NAME = "megamoe"
$REMOTE_PATH = $Sftp.remotePath
$HOST_MOUNT_ROOT = "/home/hg/yuguo"
$CONTAINER_WORKSPACE = "/workspace"
$RemotePathUnix = $REMOTE_PATH -replace '\\', '/'
$HostMountRootUnix = $HOST_MOUNT_ROOT.TrimEnd('/')
if ($RemotePathUnix -eq $HostMountRootUnix -or $RemotePathUnix.StartsWith("$HostMountRootUnix/")) {
  $CONTAINER_REPO = $RemotePathUnix -replace ('^' + [regex]::Escape($HostMountRootUnix)), $CONTAINER_WORKSPACE.TrimEnd('/')
} else {
  $REPO_NAME = Split-Path $REMOTE_PATH -Leaf
  $CONTAINER_REPO = "$($CONTAINER_WORKSPACE.TrimEnd('/'))/$REPO_NAME"
}
```

## Windows SSH Reliability Notes

On Windows, local OpenSSH config or ACLs can break authentication before the remote host is even reached.

- Prefer `ssh -F NUL ...` and `scp -F NUL ...` when you want to ignore local `~/.ssh/config`.
- If OpenSSH reports bad permissions on `~/.ssh/config`, either fix the ACLs or bypass the config with `-F NUL`.
- If OpenSSH reports bad permissions on the private key, create a temporary copy with restricted ACLs and use that copy for this session.

Example temporary-key workflow:

```powershell
$SSH_KEY_SRC = $Sftp.privateKeyPath
$SSH_KEY = Join-Path $env:TEMP "cuda-optimized-skill_id_rsa"
Copy-Item -LiteralPath $SSH_KEY_SRC -Destination $SSH_KEY -Force
icacls $SSH_KEY /inheritance:r /grant:r "$((whoami)):F"
```

## Quick Verification Workflow

1. Verify SSH login and host identity.

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "hostname && whoami && test -d /home/hg/yuguo && echo host_mount_root_ok=/home/hg/yuguo"
```

2. Verify target container exists and is running.

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker ps --format 'table {{.Names}}\t{{.Status}}' | sed -n '1,20p'"
```

If the expected container is missing from `docker ps`, check all containers before deciding it does not exist:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker ps -a --filter name=$DOCKER_NAME --format 'table {{.Names}}\t{{.Status}}'"
```

3. Verify host/container mount metadata and workspace mapping in container.

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker inspect $DOCKER_NAME --format '{{json .Mounts}}'"
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && pwd && ls -la /workspace && test -d /workspace && echo container_workspace_ok=/workspace && if [ -d $CONTAINER_REPO ]; then echo container_repo_ok=$CONTAINER_REPO; else echo container_repo_missing_not_uploaded_yet=$CONTAINER_REPO; fi && (which hipcc || true)'"
```

4. Check GPU/DCU/ROCm status, memory usage, and hardware information in container.

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && (hy-smi || rocm-smi --showuse --showmemuse || rocm-smi || /opt/dtk/bin/rocm-smi || true)'"
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && (rocninfo || rocminfo || /opt/dtk/bin/rocminfo) 2>/dev/null | egrep `"Name:|Marketing Name:|Vendor Name:|Device Type:|Compute Unit:|SIMDs per CU:|Wavefront Size:|ISA`" || true'"
```
Choose a device with low or zero compute and memory use, then use `HIP_VISIBLE_DEVICES=` to pin to that device for best performance and isolation.

If VRAM is unexpectedly high, list owning KFD PIDs and map them to host processes:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && hy-smi --showpids || true'"
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "ps -fp <pid1>,<pid2>,<pid3>"
```

5. Check Python package inventory in container.

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && pip3 list'"
```

## Compile, Test, Debug Templates

Run all project validation through the same remote execution pattern:

```powershell
# Compile check
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && python3 -m compileall .'"

# Targeted tests
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && pytest -q <path/to/test.py> -q'"

# Debug command (example)
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && python3 <your_script.py>'"

# HIP/DTK sample compile + run (example)
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && /opt/dtk/bin/hipcc -O2 hip_vector_add.cpp -o hip_vector_add && HIP_VISIBLE_DEVICES=<the device of HCU memory use 0 and HCU use 0> ./hip_vector_add'"
```

For GPU pinning, prefix the inner command with environment variables:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && HIP_VISIBLE_DEVICES=<the device of HCU memory use 0 and HCU use 0> PYTHONPATH=. pytest -q <gpu_test.py> -q'"
```

## Sync Guidance

- Keep `.vscode/sftp.json` `uploadOnSave` enabled.
- Save locally first, then execute remotely.
- Before the first upload, ensure the remote target directory exists:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "mkdir -p $REMOTE_PATH"
```

- After the first upload, verify the host path and container path both see the project:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "test -d $REMOTE_PATH && echo remote_repo_ok=$REMOTE_PATH"
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && test -d $CONTAINER_REPO && echo container_repo_ok=$CONTAINER_REPO'"
```

- If a file does not appear remotely in time, upload it explicitly:

```powershell
scp -F NUL -P $SSH_PORT -i $SSH_KEY <local_file> "${SSH_TARGET}:$REMOTE_PATH/<relative_target>"
```

- After explicit upload, verify the file exists on the host before compiling in Docker:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "ls -l $REMOTE_PATH/<relative_target>"
```

- If host-side verification passes but the file is still missing in Docker, verify the mounted path inside the container:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && cd $CONTAINER_REPO && ls -l <relative_target>'"
```

## Sync Back Container Outputs

Files generated inside `megamoe` are usually owned by `root` on the host bind mount. Downloading them from the host as `hg` works only when host permissions allow read and directory traversal.

- `root:root` files with mode `0644` and directories with mode `0755` can be pulled by `scp` or SFTP.
- `root:root` files with mode `0600` or directories with mode `0700` cannot be pulled by `hg`; expect `Permission denied`.
- Prefer fixing ownership in Docker before syncing outputs back:

```powershell
$HOST_UID_GID = ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "stat -c '%u:%g' /home/hg/yuguo"
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker exec $DOCKER_NAME bash -lc 'source /opt/dtk/env.sh && chown -R $HOST_UID_GID $CONTAINER_REPO/<output_dir>'"
scp -F NUL -P $SSH_PORT -i $SSH_KEY -r "${SSH_TARGET}:$REMOTE_PATH/<output_dir>" <local_target_dir>
```

- If ownership cannot be changed, use `source /opt/dtk/env.sh && chmod -R u+rwX,go+rX <output_dir>` in Docker as a read-only pull fallback. Avoid broad `chmod 777`.
- In PowerShell, write remote scp paths as `"${SSH_TARGET}:$REMOTE_PATH/<path>"`; `"$SSH_TARGET:$REMOTE_PATH"` is parsed incorrectly.
- Avoid creating `/workspace/cuda-optimized-skill` as root before the first upload. If the repo path is not uploaded yet, create sync-back tests under `/workspace/.codex_*` or upload the project first.

## Failure Handling

- If `docker exec` fails with "container not running", confirm status with `docker ps -a`, then start it and rerun:

```powershell
ssh -F NUL $SSH_TARGET -p $SSH_PORT -i $SSH_KEY "docker start $DOCKER_NAME"
```

- If authentication fails before reaching the remote shell, check for local OpenSSH ACL problems on `~/.ssh/config` or the private key and switch to `-F NUL` plus a temporary key copy.
- If `.vscode/sftp.json` host differs from the skill default, trust `.vscode/sftp.json`.
- If the host shell resolves to a different home prefix while `remotePath` is `/home/hg/yuguo/cuda-optimized-skill`, treat the configured `remotePath` as the source of truth and verify the file directly with `ls`.
- If `rocm-smi` is unavailable in container, run it on host once to confirm driver state, then return to Docker workflow.
- If package or import checks fail, treat environment mismatch separately from code regressions.
