# Sandbox overrides

`mcp_server_template.json` is a patched copy of the file shipped inside
`ghcr.io/scaleapi/mcp-atlas:1.2.7`, mounted read-only over the image's copy at
`/agent-environment/src/agent_environment/mcp_server_template.json`.

## Why it exists

Two of the eleven coding servers fail to start in the stock image:

    git             AttributeError: 'Server' object has no attribute 'list_tools'
    cli-mcp-server  AttributeError: 'Server' object has no attribute 'list_tools'

Both launch via `uvx`, which resolves their unpinned `mcp` dependency to 2.0.0.
That release removed the low-level `Server.list_tools` decorator both packages
are written against, so each crashes during MCP initialisation and the sandbox
registers zero tools for it.

The sandbox reports this as `Client failed to connect: Connection closed`, and
separately logs `Failed to request initial roots from client`. The roots message
is a red herring — pinning `mcp==1.12.1` fixes both servers, and `git` needs no
`--repository` argument once it starts.

## Measured impact

Stock image: 8/11 servers, and only 19 of the 55 tasks have all their gold tools
reachable. The other 36 would fail for reasons unrelated to the system under
test — the same pool contamination that depressed the 0.4.0 SR-Agents numbers.

With this patch: 11/11 servers, 127 tools, gold coverage 55/55.

## `git`'s process starts in the wrong directory

Found from a real end-to-end smoke test, not from a startup check: `git_log`
(and every other `git` tool) resolves a relative `repo_path` against
`/agent-environment`, the process's default working directory — not `/data`,
where every repo under test actually lives, and where `filesystem` and
`desktop-commander` already resolve `.` correctly. A relative path that is
correct for every other tool in the catalog (e.g. `repos/storyteller`, exactly
what `filesystem/list_directory` reports) fails against `git` with a bare
`InvalidGitRepositoryError` naming the wrong resolved path
(`/agent-environment/repos/storyteller`) — which reads as "no such repo" when
the repo is real and the path the agent used was the right one.

Verified against the live sandbox: `-r /data` (mcp-server-git's own
`--repository` flag) does not work either — `/data` itself is not a repo, only
its subdirectories are, so that flag fails at startup rather than fixing
relative-path resolution.

Fixed by launching `git` under a shell that `cd`s to `/data` first, so its
relative-path convention matches every other tool in the catalog:

    "git": { "command": "sh", "args": ["-c", "cd /data && exec uvx --with mcp==1.12.1 mcp-server-git==2026.7.10"] }

Verified end to end through the real shim after the fix: `git_log` with
`repo_path: "repos/storyteller"` returns real commit history.

This affects both arms identically — it is not gateway-specific — but was
found via the ratel arm's transcripts first while investigating a separate,
unrelated bug (`invoke_tool` receiving a stringified `params` instead of an
object, reported upstream against ratel-local, not fixed here).

## `cli-mcp-server` couldn't search inside a file

Also found from the same smoke test: `cli-mcp-server`'s `run_command` only
allowed `ls`, `cat`, `find` (its own `ALLOWED_COMMANDS` env var, upstream's
default). That's enough to list and dump a whole file, but not to search
*inside* one — the agent hit a real file (`voices.index.json`, 811,706
characters) too large to `cat` in one call, reached for `grep`, and got
`Security violation: Command 'grep' is not allowed`. With no way to search a
large file, it gave up on that path entirely rather than finding another way
in, and burned its remaining turn budget on unrelated exploration.

Added `grep` to `ALLOWED_COMMANDS`. Verified live through the real shim:
`grep -i irish /data/repos/storyteller/data/voices.11labs.json` now returns
the real matches (`"accent": "irish"`, `"accent": "american-irish"`) instead
of being refused. `ALLOW_SHELL_OPERATORS` stays `false` — no pipes, so
`cat file | grep x` still won't work, only `grep x file` directly.

## Scope of the change

`args`/`command` differ from upstream for `git` and `cli-mcp-server`; `grep`
was added to `cli-mcp-server`'s `ALLOWED_COMMANDS`. No server is added,
removed, or otherwise reconfigured.

This is a declared limitation: the environment is not byte-identical to stock
MCP-Atlas. The mode already declares non-comparability with their leaderboard
(tool exposure changed), so this does not add a new class of caveat.

Not yet checked: whether other `uvx`/relative-path-sensitive servers in the
195-tool `full` scope carry the same class of breakage. Verify before Phase 2.
