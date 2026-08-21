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

## Scope of the change

Only the `args` of `git` and `cli-mcp-server` differ from upstream, each gaining
`--with mcp==1.12.1`. No server is added, removed, or otherwise reconfigured.

This is a declared limitation: the environment is not byte-identical to stock
MCP-Atlas. The mode already declares non-comparability with their leaderboard
(tool exposure changed), so this does not add a new class of caveat.

Not yet checked: whether other `uvx` servers in the 195-tool `full` scope carry
the same breakage. Verify before Phase 2.
