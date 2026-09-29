[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        "status",
        "tools",
        "dispatch",
        "keyquery",
        "listkeys",
        "world",
        "field",
        "fieldsvg",
        "relgraph",
        "incidence",
        "semfold",
        "sandbox-list",
        "sandbox-query",
        "sandbox-violations",
        "confidence",
        "replay",
        "infer",
        "center",
        "rpc",
        "mcp"
    )]
    [string]$Action,

    [Parameter(Position = 1)]
    [string]$Query = "",

    [string]$Id = "",
    [int]$LastN = 20,
    [int]$MaxTokens = 32,
    [string]$Method = "",
    [string]$Tool = "",
    [string]$Params = "{}",
    [string]$BaseUrl = "",
    [switch]$DryRun
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

function Invoke-Bot {
    param([string[]]$CmdArgs)
    if ($DryRun) {
        Write-Host ("python " + ($CmdArgs -join " "))
        return
    }
    & python @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Invoke-BotRpcObject {
    param(
        [string]$RpcMethod,
        [hashtable]$Payload
    )
    $json = $Payload | ConvertTo-Json -Compress
    $cmd = @("bot.py", "rpc", $RpcMethod, $json)
    if ($BaseUrl) { $cmd += @("--base-url", $BaseUrl) }
    Invoke-Bot -CmdArgs $cmd
}

function Invoke-BotMcpObject {
    param(
        [string]$McpTool,
        [hashtable]$Payload
    )
    $json = $Payload | ConvertTo-Json -Compress
    $cmd = @("bot.py", "mcp", $McpTool, $json)
    if ($BaseUrl) { $cmd += @("--base-url", $BaseUrl) }
    Invoke-Bot -CmdArgs $cmd
}

Push-Location $repoRoot
try {
    switch ($Action) {
        "status" {
            Invoke-BotRpcObject "key_query" @{ query = "STATUS:" }
        }
        "tools" {
            Invoke-BotRpcObject "key_query" @{ query = "TOOLS:" }
        }
        "dispatch" {
            $payload = if ($Query) { $Query } else { "status" }
            Invoke-BotRpcObject "micronaut_dispatch" @{ query = $payload }
        }
        "keyquery" {
            $payload = if ($Query) { $Query } else { "status" }
            Invoke-BotRpcObject "key_query" @{ query = $payload }
        }
        "listkeys" {
            Invoke-BotRpcObject "list_keys" @{}
        }
        "world" {
            Invoke-BotMcpObject "world_state" @{}
        }
        "field" {
            Invoke-BotMcpObject "field_dag" @{}
        }
        "fieldsvg" {
            Invoke-BotMcpObject "field_dag_svg" @{}
        }
        "relgraph" {
            Invoke-BotMcpObject "relational_graph" @{}
        }
        "incidence" {
            Invoke-BotMcpObject "incidence_matrix" @{ include_phase_folds = $true }
        }
        "semfold" {
            $payload = @{ include_phase_folds = $true }
            if ($Query) {
                if ($Query.Contains("|")) {
                    $parts = $Query.Split("|", 2)
                    $nodeA = $parts[0].Trim()
                    $nodeB = $parts[1].Trim()
                    if ($nodeA -and $nodeB) {
                        $payload["node_a"] = $nodeA
                        $payload["node_b"] = $nodeB
                    } else {
                        $payload["node"] = $Query.Trim()
                    }
                } else {
                    $payload["node"] = $Query.Trim()
                }
            }
            Invoke-BotMcpObject "semantic_fold" $payload
        }
        "sandbox-list" {
            Invoke-BotMcpObject "sandbox_list" @{}
        }
        "sandbox-query" {
            $target = if ($Id) { $Id } else { $Query }
            if (-not $target) { throw "sandbox-query requires -Id or query argument." }
            Invoke-BotMcpObject "sandbox_query" @{ id = $target }
        }
        "sandbox-violations" {
            if ($LastN -lt 0) { throw "-LastN must be >= 0." }
            Invoke-BotMcpObject "sandbox_violations" @{ last_n = $LastN }
        }
        "confidence" {
            $target = if ($Id) { $Id } elseif ($Query) { $Query } else { "global" }
            Invoke-BotMcpObject "confidence_get" @{ id = $target }
        }
        "replay" {
            [double]$fromSeq = 0
            if ($Query) {
                [void][double]::TryParse($Query, [ref]$fromSeq)
            }
            Invoke-BotMcpObject "jrom_replay" @{ from_seq = $fromSeq }
        }
        "infer" {
            $prompt = if ($Query) { $Query } else { "status" }
            if ($MaxTokens -lt 1) { throw "-MaxTokens must be >= 1." }
            Invoke-BotRpcObject "infer" @{
                prompt = $prompt
                max_tokens = $MaxTokens
            }
        }
        "center" {
            $prompt = if ($Query) { $Query } else { "STATUS:" }
            $cmd = @("bot.py", "center", $prompt)
            if ($DryRun) {
                Write-Host ("python " + ($cmd -join " "))
            } else {
                & python @cmd
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
        }
        "rpc" {
            if (-not $Method) { throw "rpc action requires -Method." }
            $cmd = @("bot.py", "rpc", $Method, $Params)
            if ($BaseUrl) { $cmd += @("--base-url", $BaseUrl) }
            Invoke-Bot -CmdArgs $cmd
        }
        "mcp" {
            if (-not $Tool) { throw "mcp action requires -Tool." }
            $cmd = @("bot.py", "mcp", $Tool, $Params)
            if ($BaseUrl) { $cmd += @("--base-url", $BaseUrl) }
            Invoke-Bot -CmdArgs $cmd
        }
        default {
            throw "Unsupported action: $Action"
        }
    }
}
finally {
    Pop-Location
}
