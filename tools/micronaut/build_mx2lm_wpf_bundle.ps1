[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [string]$BundleName = "mx2lm-wpf-cli-v1",
    [object]$IncludeDatasetRows = $true,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "This command surface requires PowerShell 7+ (pwsh). Re-run with pwsh.exe."
}

$includeDatasetRowsBool = $true
if ($IncludeDatasetRows -is [bool]) {
    $includeDatasetRowsBool = [bool]$IncludeDatasetRows
} elseif ($IncludeDatasetRows -is [ValueType]) {
    $includeDatasetRowsBool = ([double]$IncludeDatasetRows) -ne 0
} else {
    $text = [string]$IncludeDatasetRows
    $norm = $text.Trim().ToLowerInvariant()
    if ($norm -in @("true", "1", "yes", "y", "on")) {
        $includeDatasetRowsBool = $true
    } elseif ($norm -in @("false", "0", "no", "n", "off", "")) {
        $includeDatasetRowsBool = $false
    } else {
        throw "IncludeDatasetRows must be true/false (or 1/0). Received: '$IncludeDatasetRows'"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "artifacts\micronaut-bundles"
}

$bundleRoot = Join-Path $OutputRoot $BundleName
if (Test-Path $bundleRoot) {
    if (-not $Force) {
        throw "Target bundle already exists: $bundleRoot. Use -Force to overwrite."
    }
    Remove-Item -LiteralPath $bundleRoot -Recurse -Force
}

$null = New-Item -ItemType Directory -Path $bundleRoot -Force

$generated = New-Object System.Collections.Generic.List[string]
$copied = New-Object System.Collections.Generic.List[object]

function Write-GeneratedFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $absolutePath = Join-Path $bundleRoot $RelativePath
    $parent = Split-Path $absolutePath -Parent
    $null = New-Item -ItemType Directory -Path $parent -Force
    $body = $Content.TrimStart([char]13, [char]10)
    Set-Content -LiteralPath $absolutePath -Value $body -Encoding UTF8
    $generated.Add($RelativePath) | Out-Null
}

function Mirror-SourceFile {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$TargetRelativePath
    )
    if (-not (Test-Path $SourcePath)) {
        return
    }
    $destination = Join-Path $bundleRoot $TargetRelativePath
    $null = New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force
    Copy-Item -LiteralPath $SourcePath -Destination $destination -Force
    $copied.Add([ordered]@{
            source = $SourcePath
            target = $TargetRelativePath
        }) | Out-Null
}

function Mirror-DirectoryFiles {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDirectory,
        [Parameter(Mandatory = $true)][string]$TargetRelativeDirectory,
        [string]$Filter = "*"
    )
    if (-not (Test-Path $SourceDirectory)) {
        return
    }
    Get-ChildItem -Path $SourceDirectory -File -Recurse -Filter $Filter |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($SourceDirectory.Length).TrimStart('\')
            $targetRel = Join-Path $TargetRelativeDirectory $relative
            Mirror-SourceFile -SourcePath $_.FullName -TargetRelativePath $targetRel
        }
}

function New-BundleDuplicateInventory {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [string[]]$ExcludeRelativePaths = @()
    )

    $files = Get-ChildItem -LiteralPath $RootPath -File -Recurse | Sort-Object FullName
    $rows = New-Object System.Collections.Generic.List[object]

    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($RootPath.Length).TrimStart('\')
        if ($ExcludeRelativePaths -contains $relativePath) {
            continue
        }
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $rows.Add([pscustomobject][ordered]@{
                path = $relativePath
                size_bytes = [int64]$file.Length
                sha256 = $hash
            }) | Out-Null
    }

    $groups = @(
        $rows |
            Group-Object -Property sha256 |
            Where-Object { $_.Count -gt 1 } |
            Sort-Object -Property @{Expression = "Count"; Descending = $true }, @{Expression = "Name"; Descending = $false }
    )

    $duplicateGroups = New-Object System.Collections.Generic.List[object]
    foreach ($group in $groups) {
        $groupPaths = @($group.Group | Sort-Object path | ForEach-Object { $_.path })
        $sample = $group.Group | Select-Object -First 1
        $duplicateGroups.Add([ordered]@{
                sha256 = $group.Name
                count = [int]$group.Count
                size_bytes = [int64]$sample.size_bytes
                paths = $groupPaths
            }) | Out-Null
    }

    $duplicateFileCount = 0
    if ($duplicateGroups.Count -gt 0) {
        $duplicateFileCount = [int](($duplicateGroups | Measure-Object -Property count -Sum).Sum)
    }

    return [ordered]@{
        schema = "mx2.bundle.duplicate-files.v1"
        generated_utc = $utcNow
        bundle = $BundleName
        target_root = $RootPath
        file_count = [int]$rows.Count
        duplicate_group_count = [int]$duplicateGroups.Count
        duplicate_file_count = $duplicateFileCount
        duplicate_groups = [object[]]$duplicateGroups
    }
}

# Mirror a few local stack surfaces as read-only context.
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "openai.yaml") -TargetRelativePath "source-mirror\openai.source.yaml"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "bot.py") -TargetRelativePath "source-mirror\bot.source.py"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "tools\micronaut\kmu.ps1") -TargetRelativePath "source-mirror\kmu.source.ps1"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\alice\mx2lm.aiml") -TargetRelativePath "source-mirror\mx2lm.source.aiml"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\alice\greetings.aiml") -TargetRelativePath "source-mirror\greetings.source.aiml"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\alice\advisor.aiml") -TargetRelativePath "source-mirror\advisor.source.aiml"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\alice\plan.aiml") -TargetRelativePath "source-mirror\plan.source.aiml"

# The KHANARY runtime loads AIML from <data_dir>/alice (AIMLEngine::load_dir
# returns 0 when that directory is absent — there is no fallback path), while
# the WPF command surface reads <data_dir>/aiml. Ship the complete ALICE set in
# both layouts so neither consumer is left without its categories.
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\alice") -TargetRelativeDirectory "data\alice" -Filter "*.aiml"
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\alice") -TargetRelativeDirectory "data\aiml" -Filter "*.aiml"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "asx-atomic-block.png") -TargetRelativePath "cli\wpf\assets\asx-atomic-block.png"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "asx-atomic-blocks.png") -TargetRelativePath "cli\wpf\assets\asx-atomic-blocks.png"
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\schema") -TargetRelativeDirectory "data\schema" -Filter "*.json"
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\tracks") -TargetRelativeDirectory "data\tracks" -Filter "*.json"
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\manifest") -TargetRelativeDirectory "data\manifest" -Filter "*.json"
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\mx2lm") -TargetRelativeDirectory "data\mx2lm"

# Grammar surfaces the runtime reads directly at startup
# (data/grammar/greeting-bank.json + the runtime EBNF specs). The dictionary
# lexicon/relation JSONL corpora in data\grammar are tens of MB of build-time
# input and are deliberately NOT shipped — mirror the runtime-read surfaces only.
Mirror-DirectoryFiles -SourceDirectory (Join-Path $repoRoot "data\grammar") -TargetRelativeDirectory "data\grammar" -Filter "*.ebnf"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\greeting-bank.json") -TargetRelativePath "data\grammar\greeting-bank.json"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\advisor-bootstrap.json") -TargetRelativePath "data\grammar\advisor-bootstrap.json"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\plan-bootstrap.json") -TargetRelativePath "data\grammar\plan-bootstrap.json"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\control-flow.key-grammar.jsonl") -TargetRelativePath "data\grammar\control-flow.key-grammar.jsonl"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\greeting-bank.key-grammar.jsonl") -TargetRelativePath "data\grammar\greeting-bank.key-grammar.jsonl"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\advisor-bootstrap.key-grammar.jsonl") -TargetRelativePath "data\grammar\advisor-bootstrap.key-grammar.jsonl"
Mirror-SourceFile -SourcePath (Join-Path $repoRoot "data\grammar\plan-bootstrap.key-grammar.jsonl") -TargetRelativePath "data\grammar\plan-bootstrap.key-grammar.jsonl"

$utcNow = (Get-Date).ToUniversalTime().ToString("o")
$defaultGgufModel = "E:\models\GPT2\gemma_toolcall\gemma_chat_toolcall_q8.gguf"
$defaultDdsRoot = "E:\models\GPT-DDS\GPT-OSS"
$defaultDdsKson = Join-Path $defaultDdsRoot "gpt_oss_20b_folds.kson"
$defaultDdsManifest = Join-Path $defaultDdsRoot "folds\model_dds_manifest.json"
$defaultDdsShards = Join-Path $defaultDdsRoot "folds\dds_shards"

$bundleToolsManifestPath = Join-Path $bundleRoot "data\manifest\tools.manifest.json"
if (-not (Test-Path $bundleToolsManifestPath)) {
    $toolsManifest = [ordered]@{
        schema = "tools.manifest.v1"
        generated_utc = $utcNow
        source = "bundle-generated-fallback"
        rpc = @(
            "key_query",
            "micronaut_dispatch",
            "fold_query",
            "list_keys",
            "key_encode",
            "key_decode",
            "keyspace_status",
            "infer",
            "resolve_manifest"
        )
        mcp = @(
            "world_state",
            "field_dag",
            "field_dag_svg",
            "relational_graph",
            "incidence_matrix",
            "semantic_fold",
            "confidence_get",
            "jrom_replay",
            "sandbox_query",
            "sandbox_list",
            "sandbox_violations"
        )
    }
    $null = New-Item -ItemType Directory -Path (Split-Path $bundleToolsManifestPath -Parent) -Force
    ($toolsManifest | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $bundleToolsManifestPath -Encoding UTF8
    $generated.Add("data\manifest\tools.manifest.json") | Out-Null
}

Write-GeneratedFile -RelativePath "MICRONAUTS.md" -Content @"
# $BundleName

MX-2 style micronaut bundle generated deterministically from local KHANARY stack surfaces.

- Source policy: repo-local only (no `public_html` dependency)
- Generated at: $utcNow
- Bundle root: `$(Resolve-Path $bundleRoot)`

## Layout

- `folds.toml`
- `micronaut.registry.xjson`
- `model.runtime.json`
- `bots.py`
- `agents/`
- `evolution/`
- `cli/wpf/`
- `scripts/`
- `dataset/` (optional rows)
- `data/schema/`, `data/tracks/`, `data/manifest/`, `data/mx2lm/`
- `data/aiml/`
- `source-mirror/` (read-only copied references)
- `bundle_duplicate_files.json` (content-hash duplicate inventory)
"@

Write-GeneratedFile -RelativePath "folds.toml" -Content @"
[bundle]
name = "$BundleName"
layout = "mx2"
schema = "mx2.layout.v1"
generated_utc = "$utcNow"

[paths]
registry = "micronaut.registry.xjson"
runtime = "model.runtime.json"
bots = "bots.py"
agents = "agents"
evolution = "evolution"
wpf_cli = "cli/wpf/KhanaryMicronautCli.Wpf.csproj"
primary_track = "data/tracks/TOOL-u.semantic-tracks.v1.json"
mx2lm_track = "data/tracks/Mx2LM.semantic-tracks.v1.json"
aiml_file = "data/aiml/mx2lm.aiml"

[[fold]]
id = "aiml-runtime"
phase = "Pop"
op = "key_query"
query = "Mx2LM: status"
delegate = "Mx2LM"

[[fold]]
id = "status"
phase = "Pop"
op = "key_query"
query = "STATUS:"
delegate = "TOOL-u"

[[fold]]
id = "dispatch"
phase = "Wo"
op = "micronaut_dispatch"
query = "status"
delegate = "ORCH-u"

[[fold]]
id = "semantic-fold"
phase = "Yax"
op = "semantic_fold"
delegate = "REASON-u"

[[fold]]
id = "verify"
phase = "Chen"
op = "confidence_get"
query = "global"
delegate = "INFER-u"

[[fold]]
id = "commit"
phase = "Xul"
op = "jrom_replay"
query = "0"
delegate = "MEM-u"
"@

$registryObject = [ordered]@{
    "@schema" = "micronaut.registry.xjson.v1"
    bundle = [ordered]@{
        name = $BundleName
        generated_utc = $utcNow
        layout = "mx2"
        cli = "wpf-terminal-console"
    }
    tracks = @(
        [ordered]@{ id = "TOOL-u"; role = "tool-planning"; active = $true },
        [ordered]@{ id = "Mx2LM"; role = "aiml-runtime"; active = $true },
        [ordered]@{ id = "ORCH-u"; role = "routing"; active = $true },
        [ordered]@{ id = "REASON-u"; role = "semantic-fold"; active = $true },
        [ordered]@{ id = "INFER-u"; role = "confidence"; active = $true },
        [ordered]@{ id = "MEM-u"; role = "replay"; active = $true },
        [ordered]@{ id = "GREETING-u"; role = "salutation-style"; active = $true },
        [ordered]@{ id = "CONTROL-u"; role = "control-flow"; active = $true },
        [ordered]@{ id = "ADVISOR-µ"; role = "expert-admission-and-proposal"; active = $true }
        [ordered]@{ id = "PLAN-µ"; role = "task-planning"; active = $true }
        [ordered]@{ id = "MEMORY-µ"; role = "shared-context-and-distillation"; active = $true }
    )
    routes = [ordered]@{
        aiml = "Mx2LM"
        status = "TOOL-u"
        tools = "TOOL-u"
        dispatch = "ORCH-u"
        semfold = "REASON-u"
        confidence = "INFER-u"
        replay = "MEM-u"
        advisor = "ADVISOR-µ"
        plan = "PLAN-µ"
        memory = "MEMORY-µ"
    }
    surfaces = [ordered]@{
        command_shim = "agents/commands.ps1"
        bots = "bots.py"
        wpf_project = "cli/wpf/KhanaryMicronautCli.Wpf.csproj"
        tools_manifest = "data/manifest/tools.manifest.json"
        advisor_card = "data/tracks/ADVISOR-u.semantic-tracks.v1.json#ui"
        plan_card = "data/tracks/PLAN-u.semantic-tracks.v1.json#ui"
        memory_card = "data/tracks/MEMORY-u.semantic-tracks.v1.json#ui_card_schema"
    }
    seeds = [ordered]@{
        primary_track = "data/tracks/TOOL-u.semantic-tracks.v1.json"
        mx2lm_track = "data/tracks/Mx2LM.semantic-tracks.v1.json"
        greeting_track = "data/tracks/GREETING-u.semantic-tracks.v1.json"
        control_track = "data/tracks/CONTROL-u.semantic-tracks.v1.json"
        advisor_track = "data/tracks/ADVISOR-u.semantic-tracks.v1.json"
        plan_track = "data/tracks/PLAN-u.semantic-tracks.v1.json"
        memory_track = "data/tracks/MEMORY-u.semantic-tracks.v1.json"
    }
}
Write-GeneratedFile -RelativePath "micronaut.registry.xjson" -Content ($registryObject | ConvertTo-Json -Depth 8)

$runtimeObject = [ordered]@{
    schema = "model.runtime.v1"
    bundle = $BundleName
    generated_utc = $utcNow
    paths = [ordered]@{
        registry = "./micronaut.registry.xjson"
        folds = "./folds.toml"
        bots = "./bots.py"
        agents = "./agents/openai.yaml"
        command_shim = "./agents/commands.ps1"
        evolution_runtime = "./evolution/engine.runtime.json"
        wpf_project = "./cli/wpf/KhanaryMicronautCli.Wpf.csproj"
        wpf_launcher = "./scripts/launch-wpf-cli.ps1"
        schema_dir = "./data/schema"
        tracks_dir = "./data/tracks"
        primary_track = "./data/tracks/TOOL-u.semantic-tracks.v1.json"
        mx2lm_track = "./data/tracks/Mx2LM.semantic-tracks.v1.json"
        aiml_file = "./data/aiml/mx2lm.aiml"
        advisor_aiml_file = "./data/aiml/advisor.aiml"
        advisor_source = "./data/grammar/advisor-bootstrap.json"
        plan_aiml_file = "./data/aiml/plan.aiml"
        plan_source = "./data/grammar/plan-bootstrap.json"
        manifest_dir = "./data/manifest"
        tools_manifest = "./data/manifest/tools.manifest.json"
        mx2lm_dir = "./data/mx2lm"
        tool_verb_ngrams = "./data/mx2lm/tool_verb_ngrams.json"
    }
    defaults = [ordered]@{
        action = "aiml"
        query = "status"
        gguf_model = $defaultGgufModel
        dds_kson = $defaultDdsKson
        dds_manifest = $defaultDdsManifest
        dds_shards = $defaultDdsShards
    }
    policy = [ordered]@{
        source_locked_readonly = $true
        source_root = "repo-local-surfaces"
        rewrite_to_self = $true
    }
}
Write-GeneratedFile -RelativePath "model.runtime.json" -Content ($runtimeObject | ConvertTo-Json -Depth 8)

Write-GeneratedFile -RelativePath "bots.py" -Content @'
from __future__ import annotations

import argparse
import base64
import subprocess
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
COMMAND_SHIM = BUNDLE_ROOT / "agents" / "commands.ps1"


def run_action(action: str, query: str = "", raw_json: bool = True) -> str:
    encoded = base64.b64encode(query.encode("utf-8")).decode("ascii") if query else ""
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(COMMAND_SHIM),
        "-Action",
        action,
    ]
    if encoded:
        cmd.extend(["-QueryBase64", encoded])
    if raw_json:
        cmd.append("-RawJson")

    proc = subprocess.run(
        cmd,
        cwd=str(BUNDLE_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"action failed with code {proc.returncode}")
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="MX2LM WPF bundle bot shim")
    parser.add_argument("action", nargs="?", default="aiml")
    parser.add_argument("query", nargs="?", default="status")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    output = run_action(args.action, args.query, raw_json=(args.raw or True))
    print(output.rstrip())


if __name__ == "__main__":
    main()
'@

Write-GeneratedFile -RelativePath "agents\openai.yaml" -Content @"
display_name: MX2LM WPF Bundle
short_description: Terminal-plus-console style WPF CLI wrapper around KHANARY micronaut RPC/MCP surfaces.
profile:
  runtime: khanary
  model: local
  mode: command-center

endpoints:
  rpc: http://127.0.0.1:8787
  mcp: http://127.0.0.1:8764

entrypoints:
  command_shim: agents/commands.ps1
  bots: bots.py
  wpf_launcher: scripts/launch-wpf-cli.ps1
"@

Write-GeneratedFile -RelativePath "agents\commands.ps1" -Content @'
[CmdletBinding()]
param(
    [ValidateSet("aiml", "status", "tools", "keyquery", "dispatch", "center", "rpc", "mcp", "advisor_consult")]
    [string]$Action = "aiml",
    [string]$Query = "",
    [string]$QueryBase64 = "",
    [string]$Method = "",
    [string]$Tool = "",
    [string]$ParamsJson = "{}",
    [string]$RpcUrl = "http://127.0.0.1:8787",
    [string]$McpUrl = "http://127.0.0.1:8764",
    [string]$AdvisorUrl = "http://127.0.0.1:17887",
    [string]$ChatMuUrl  = "http://127.0.0.1:17889",
    [string]$ToolMuUrl  = "http://127.0.0.1:17890",
    [string]$KhanaryExe = "",
    [string]$DataDir = "",
    [string]$GgufModel = "",
    [string]$DdsKson = "",
    [string]$DdsManifest = "",
    [string]$DdsShards = "",
    [int]$StartupTimeoutSec = 120,
    [switch]$RawJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "This command surface requires PowerShell 7+ (pwsh). Re-run with pwsh.exe."
}
$defaultGgufModel = ""
$defaultDdsKson = ""
$defaultDdsManifest = ""
$defaultDdsShards = ""

if ([string]::IsNullOrWhiteSpace($Query) -and -not [string]::IsNullOrWhiteSpace($QueryBase64)) {
    $bytes = [Convert]::FromBase64String($QueryBase64)
    $Query = [Text.Encoding]::UTF8.GetString($bytes)
}

$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $bundleRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $repoRoot "data"
}
if ([string]::IsNullOrWhiteSpace($GgufModel)) {
    if (-not [string]::IsNullOrWhiteSpace($env:KHANARY_GGUF_MODEL)) {
        $GgufModel = $env:KHANARY_GGUF_MODEL
    } else {
        $GgufModel = $defaultGgufModel
    }
}
if ([string]::IsNullOrWhiteSpace($DdsKson)) {
    if (-not [string]::IsNullOrWhiteSpace($env:KHANARY_DDS_KSON)) {
        $DdsKson = $env:KHANARY_DDS_KSON
    } else {
        $DdsKson = $defaultDdsKson
    }
}
if ([string]::IsNullOrWhiteSpace($DdsManifest)) {
    if (-not [string]::IsNullOrWhiteSpace($env:KHANARY_DDS_MANIFEST)) {
        $DdsManifest = $env:KHANARY_DDS_MANIFEST
    } else {
        $DdsManifest = $defaultDdsManifest
    }
}
if ([string]::IsNullOrWhiteSpace($DdsShards)) {
    if (-not [string]::IsNullOrWhiteSpace($env:KHANARY_DDS_SHARDS)) {
        $DdsShards = $env:KHANARY_DDS_SHARDS
    } else {
        $DdsShards = $defaultDdsShards
    }
}
$aimlFile = Join-Path $bundleRoot "data\aiml\mx2lm.aiml"
$toolVerbMapFile = Join-Path $bundleRoot "data\mx2lm\tool_verb_ngrams.json"

function Test-RpcReady {
    try {
        $body = @{
            jsonrpc = "2.0"
            id = 1
            method = "key_query"
            params = @{ query = "STATUS:" }
        } | ConvertTo-Json -Depth 6 -Compress
        $resp = Invoke-RestMethod -Uri $RpcUrl -Method Post -ContentType "application/json" -Body $body -TimeoutSec 2
        if ($null -eq $resp) {
            return $false
        }
        if ($null -ne $resp.PSObject -and $resp.PSObject.Properties.Match("error").Count -gt 0) {
            return $false
        }
        return $true
    } catch {
        return $false
    }
}

function Resolve-PortFromUrl {
    param(
        [string]$Url,
        [int]$FallbackPort
    )
    if (-not [string]::IsNullOrWhiteSpace($Url)) {
        try {
            $uri = [Uri]$Url
            if ($uri.Port -gt 0) {
                return [string]$uri.Port
            }
        } catch {
            # ignore parse errors and fall back
        }
    }
    return [string]$FallbackPort
}

function Resolve-KhanaryExePath {
    if (-not [string]::IsNullOrWhiteSpace($KhanaryExe) -and (Test-Path $KhanaryExe)) {
        return (Resolve-Path $KhanaryExe).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:KHANARY_EXE) -and (Test-Path $env:KHANARY_EXE)) {
        return (Resolve-Path $env:KHANARY_EXE).Path
    }
    $candidates = @(
        (Join-Path $repoRoot "build\Release\KHANARY.exe"),
        (Join-Path $repoRoot "build\Release\Release\KHANARY.exe"),
        (Join-Path $repoRoot "build\RelWithDebInfo\KHANARY.exe"),
        (Join-Path $repoRoot "KHANARY.exe"),
        (Join-Path $repoRoot "build\Debug\KHANARY.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Ensure-KhanaryRuntime {
    if (Test-RpcReady) {
        return
    }

    $exePath = Resolve-KhanaryExePath
    if ([string]::IsNullOrWhiteSpace($exePath)) {
        throw "KHANARY.exe not found. Set -KhanaryExe or KHANARY_EXE."
    }
    if (-not (Test-Path $DataDir)) {
        throw "Data directory not found: $DataDir"
    }

    $workDir = Split-Path $exePath -Parent
    $rpcPort = Resolve-PortFromUrl -Url $RpcUrl -FallbackPort 8787
    $mcpPort = Resolve-PortFromUrl -Url $McpUrl -FallbackPort 8764
    $args = @(
        "--data-dir", $DataDir,
        "--port-rpc", $rpcPort,
        "--port-mcp", $mcpPort
    )
    if ((Test-Path $DdsKson) -and (Test-Path $DdsManifest) -and (Test-Path $DdsShards)) {
        $args += @("--dds-kson", $DdsKson, "--dds-manifest", $DdsManifest, "--dds-shards", $DdsShards)
    }
    if (-not [string]::IsNullOrWhiteSpace($GgufModel) -and (Test-Path $GgufModel)) {
        $args += @("--gguf-model", $GgufModel)
    }
    $startedProcess = $null
    $existingProcess = Get-Process -Name KHANARY -ErrorAction SilentlyContinue |
        Where-Object {
            try {
                $_.Path -eq $exePath
            } catch {
                $false
            }
        } |
        Select-Object -First 1

    if ($null -eq $existingProcess) {
        $startedProcess = Start-Process -FilePath $exePath -ArgumentList $args -WorkingDirectory $workDir -WindowStyle Hidden -PassThru
    } else {
        $startedProcess = $existingProcess
    }

    $deadline = (Get-Date).AddSeconds([Math]::Max(3, $StartupTimeoutSec))
    do {
        Start-Sleep -Milliseconds 500
        if ($null -ne $startedProcess -and $startedProcess.HasExited) {
            throw "KHANARY runtime exited before ready (exit code $($startedProcess.ExitCode))."
        }
        if (Test-RpcReady) {
            return
        }
    } while ((Get-Date) -lt $deadline)

    throw "KHANARY runtime did not become ready at $RpcUrl in $StartupTimeoutSec seconds."
}

function Parse-ParamsJson {
    param([string]$JsonText)
    if ([string]::IsNullOrWhiteSpace($JsonText)) {
        return @{}
    }
    return (ConvertFrom-Json -InputObject $JsonText -AsHashtable)
}

function Resolve-AimlToolRoute {
    param([string]$InputText)
    if ([string]::IsNullOrWhiteSpace($InputText)) {
        return $null
    }
    if (-not (Test-Path $toolVerbMapFile)) {
        return $null
    }

    $cfg = Get-Content -LiteralPath $toolVerbMapFile -Raw | ConvertFrom-Json -AsHashtable
    $rows = $cfg["tool_verb_ngrams"]
    if ($null -eq $rows) {
        return $null
    }

    $lower = $InputText.ToLowerInvariant()
    foreach ($row in $rows) {
        if ($null -eq $row) { continue }
        $ngrams = @($row["ngrams"])
        foreach ($ng in $ngrams) {
            if ([string]::IsNullOrWhiteSpace([string]$ng)) { continue }
            if ($lower.Contains(([string]$ng).ToLowerInvariant())) {
                return $row
            }
        }
    }
    return $null
}

function Invoke-JsonRpcCall {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$MethodName,
        [Parameter(Mandatory = $true)][object]$ParamsObject
    )

    Ensure-KhanaryRuntime

    $payload = @{
        jsonrpc = "2.0"
        id = [int](Get-Random -Minimum 1000 -Maximum 999999)
        method = $MethodName
        params = $ParamsObject
    } | ConvertTo-Json -Depth 20 -Compress

    $resp = Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 90
    $respError = $null
    if ($null -ne $resp -and $null -ne $resp.PSObject -and $resp.PSObject.Properties.Match("error").Count -gt 0) {
        $respError = $resp.error
    }
    if ($null -ne $respError) {
        $errMsg = $respError.message
        if ([string]::IsNullOrWhiteSpace($errMsg)) {
            $errMsg = ($respError | ConvertTo-Json -Depth 10 -Compress)
        }
        throw "RPC error calling '$MethodName': $errMsg"
    }
    return $resp
}

function Emit-Response {
    param([object]$Response)
    if ($RawJson) {
        $Response | ConvertTo-Json -Depth 20
    } else {
        if ($null -ne $Response -and $null -ne $Response.PSObject -and $Response.PSObject.Properties.Match("result").Count -gt 0) {
            $Response.result | ConvertTo-Json -Depth 20
        } else {
            $Response | ConvertTo-Json -Depth 20
        }
    }
}

function Emit-ResultPayload {
    param([object]$ResultPayload)
    if ($RawJson) {
        [ordered]@{
            jsonrpc = "2.0"
            result = $ResultPayload
        } | ConvertTo-Json -Depth 20
    } else {
        $ResultPayload | ConvertTo-Json -Depth 20
    }
}

function Read-ObjectMember {
    param(
        [object]$Object,
        [string]$Name
    )
    if ($null -eq $Object -or [string]::IsNullOrWhiteSpace($Name)) {
        return $null
    }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) {
            return $Object[$Name]
        }
        return $null
    }
    if ($null -eq $Object.PSObject) {
        return $null
    }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) {
        return $null
    }
    return $prop.Value
}

function Read-ObjectString {
    param(
        [object]$Object,
        [string]$Name,
        [string]$Default = ""
    )
    $value = Read-ObjectMember -Object $Object -Name $Name
    if ($null -eq $value) {
        return $Default
    }
    $text = [string]$value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $Default
    }
    return $text.Trim()
}

function Read-ObjectBool {
    param(
        [object]$Object,
        [string]$Name,
        [bool]$Default = $false
    )
    $value = Read-ObjectMember -Object $Object -Name $Name
    if ($null -eq $value) {
        return $Default
    }
    if ($value -is [bool]) {
        return $value
    }
    $text = [string]$value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $Default
    }
    $parsed = $Default
    if ([bool]::TryParse($text, [ref]$parsed)) {
        return $parsed
    }
    return $Default
}

function Resolve-WebResearchQuery {
    param(
        [string]$InputText,
        [string]$FallbackText = "latest news"
    )
    $q = if ([string]::IsNullOrWhiteSpace($InputText)) { "" } else { $InputText.Trim() }
    $q = $q -replace '(?i)^\s*@mx2lm\s*:?\s*', ''
    $q = $q -replace '(?i)^\s*mx2lm\s*:\s*', ''
    $q = $q -replace '(?i)^\s*(search( the)? web|web research)\s*', ''
    $q = $q -replace '(?i)^\s*(for|about|on)\s+', ''
    $q = $q -replace '(?i)^\s*(recent|latest)\s+news\b', ''
    $q = $q -replace '(?i)^\s*headlines?\b', ''
    $q = $q -replace '(?i)^\s*news\b', ''
    $q = $q -replace '(?i)^\s*(for|about|on)\s+', ''
    $q = $q -replace '^\s*[:\-]\s*', ''
    $q = $q.Trim()
    if ([string]::IsNullOrWhiteSpace($q)) {
        $q = if ([string]::IsNullOrWhiteSpace($FallbackText)) { "latest news" } else { $FallbackText.Trim() }
    }
    if ([string]::IsNullOrWhiteSpace($q)) {
        return "latest news"
    }
    return $q
}

function Invoke-WebResearchFetch {
    param(
        [string]$SearchText,
        [int]$MaxResults = 5
    )

    $query = Resolve-WebResearchQuery -InputText $SearchText -FallbackText "latest news"
    $encoded = [Uri]::EscapeDataString($query)
    $urls = @(
        "https://news.google.com/rss/search?q=$encoded&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    )

    $lastErr = ""
    foreach ($url in $urls) {
        try {
            $resp = Invoke-WebRequest -Uri $url -Method Get -Headers @{ "User-Agent" = "KHANARY-WPF-CLI/1.0" } -TimeoutSec 20
            $content = if ($null -ne $resp -and $null -ne $resp.Content) { [string]$resp.Content } else { "" }
            if ([string]::IsNullOrWhiteSpace($content)) {
                continue
            }
            [xml]$rss = $content
            $items = @()
            if ($null -ne $rss -and $null -ne $rss.rss -and $null -ne $rss.rss.channel -and $null -ne $rss.rss.channel.item) {
                $items = @($rss.rss.channel.item)
            } elseif ($null -ne $rss -and $null -ne $rss.feed -and $null -ne $rss.feed.entry) {
                $items = @($rss.feed.entry)
            }
            if ($items.Count -eq 0) {
                continue
            }

            $rows = @()
            foreach ($item in ($items | Select-Object -First ([Math]::Max(1, $MaxResults)))) {
                $title = ""
                $link = ""
                $publishedRaw = ""
                $sourceName = ""
                if ($item -is [System.Xml.XmlElement]) {
                    $titleNode = $item.SelectSingleNode("title")
                    if ($null -ne $titleNode) { $title = [string]$titleNode.InnerText }
                    $linkNode = $item.SelectSingleNode("link")
                    if ($null -ne $linkNode) {
                        if ($null -ne $linkNode.Attributes -and $null -ne $linkNode.Attributes["href"]) {
                            $link = [string]$linkNode.Attributes["href"].Value
                        } else {
                            $link = [string]$linkNode.InnerText
                        }
                    }
                    $publishedNode = $item.SelectSingleNode("pubDate")
                    if ($null -eq $publishedNode) { $publishedNode = $item.SelectSingleNode("updated") }
                    if ($null -eq $publishedNode) { $publishedNode = $item.SelectSingleNode("published") }
                    if ($null -ne $publishedNode) { $publishedRaw = [string]$publishedNode.InnerText }
                    $sourceNode = $item.SelectSingleNode("source")
                    if ($null -ne $sourceNode) { $sourceName = [string]$sourceNode.InnerText }
                } else {
                    $title = Read-ObjectString -Object $item -Name "title"
                    $link = Read-ObjectString -Object $item -Name "link"
                    $publishedRaw = Read-ObjectString -Object $item -Name "pubDate"
                    $sourceObj = Read-ObjectMember -Object $item -Name "source"
                    if ($sourceObj -is [string]) {
                        $sourceName = [string]$sourceObj
                    } elseif ($null -ne $sourceObj) {
                        $sourceName = Read-ObjectString -Object $sourceObj -Name "#text"
                        if ([string]::IsNullOrWhiteSpace($sourceName)) {
                            $sourceName = [string]$sourceObj
                        }
                    }
                }
                if ([string]::IsNullOrWhiteSpace($title)) {
                    $title = "Untitled result"
                }
                $title = [System.Net.WebUtility]::HtmlDecode($title).Trim()

                $publishedUtc = ""
                if (-not [string]::IsNullOrWhiteSpace($publishedRaw)) {
                    try {
                        $publishedUtc = (Get-Date $publishedRaw).ToUniversalTime().ToString("yyyy-MM-dd HH:mm 'UTC'")
                    } catch {
                        $publishedUtc = $publishedRaw
                    }
                }
                if (-not [string]::IsNullOrWhiteSpace($sourceName)) {
                    $sourceName = [System.Net.WebUtility]::HtmlDecode($sourceName).Trim()
                }

                $rows += [ordered]@{
                    title = $title
                    url = $link
                    source = $sourceName
                    published_utc = $publishedUtc
                }
            }

            if ($rows.Count -gt 0) {
                return [ordered]@{
                    ok = $true
                    provider = "google-news-rss"
                    source_url = $url
                    query = $query
                    results = $rows
                }
            }
        } catch {
            $lastErr = $_.Exception.Message
        }
    }

    if ([string]::IsNullOrWhiteSpace($lastErr)) {
        $lastErr = "No web headlines were returned."
    }
    return [ordered]@{
        ok = $false
        provider = "google-news-rss"
        source_url = ""
        query = $query
        error = $lastErr
        results = @()
    }
}

function Build-WebResearchResponse {
    param(
        [string]$InputText,
        [object]$PlanPayload
    )

    $planQuery = Read-ObjectString -Object $PlanPayload -Name "query"
    if ([string]::IsNullOrWhiteSpace($planQuery)) {
        $planResearch = Read-ObjectMember -Object $PlanPayload -Name "research"
        $planQuery = Read-ObjectString -Object $planResearch -Name "query"
    }
    $queryCandidate = if ([string]::IsNullOrWhiteSpace($planQuery)) { $InputText } else { $planQuery }
    $normalizedQuery = Resolve-WebResearchQuery -InputText $queryCandidate -FallbackText "latest news"
    $fetch = Invoke-WebResearchFetch -SearchText $normalizedQuery -MaxResults 5
    $results = @()
    $fetchedResults = Read-ObjectMember -Object $fetch -Name "results"
    if ($null -ne $fetchedResults) {
        $results = @($fetchedResults)
    }

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("Top web results for: $normalizedQuery")
    if ($results.Count -eq 0) {
        $lines.Add("No results were returned right now.")
        $fetchErr = Read-ObjectString -Object $fetch -Name "error"
        if (-not [string]::IsNullOrWhiteSpace($fetchErr)) {
            $lines.Add("Reason: $fetchErr")
        }
    } else {
        $i = 1
        foreach ($row in $results) {
            $title = Read-ObjectString -Object $row -Name "title" -Default "Untitled result"
            $url = Read-ObjectString -Object $row -Name "url"
            $sourceName = Read-ObjectString -Object $row -Name "source"
            $publishedUtc = Read-ObjectString -Object $row -Name "published_utc"
            $meta = @()
            if (-not [string]::IsNullOrWhiteSpace($sourceName)) { $meta += $sourceName }
            if (-not [string]::IsNullOrWhiteSpace($publishedUtc)) { $meta += $publishedUtc }
            if ($meta.Count -gt 0) {
                $lines.Add("$i. $title — $($meta -join ' | ')")
            } else {
                $lines.Add("$i. $title")
            }
            if (-not [string]::IsNullOrWhiteSpace($url)) {
                $lines.Add("   $url")
            }
            $i++
        }
    }
    $text = [string]::Join([Environment]::NewLine, $lines)

    $status = if ($results.Count -gt 0) { "research_results_ready" } else { "research_results_unavailable" }
    $planPhase = Read-ObjectString -Object $PlanPayload -Name "response_phase" -Default "Pop.research_dispatch"
    $responsePhase = if ($status -eq "research_results_ready") { "Sek.web_retrieval" } else { $planPhase }
    $governingTrack = Read-ObjectString -Object $PlanPayload -Name "governing_track" -Default "WEB-RESEARCH-µ"
    $trackReason = Read-ObjectString -Object $PlanPayload -Name "track_reason" -Default "live_web_retrieval"
    $trackGoverned = Read-ObjectBool -Object $PlanPayload -Name "track_governed" -Default $true

    return [ordered]@{
        ok = $true
        source = "web-research"
        status = $status
        text = $text
        response = $text
        query = $normalizedQuery
        governing_track = $governingTrack
        response_phase = $responsePhase
        track_governed = $trackGoverned
        track_reason = $trackReason
        research = [ordered]@{
            mode = "live-web-search"
            provider = Read-ObjectString -Object $fetch -Name "provider" -Default "google-news-rss"
            source_url = Read-ObjectString -Object $fetch -Name "source_url"
            citation_required = $true
            fetched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            result_count = $results.Count
        }
        results = $results
    }
}

function Invoke-AdvisorQuery {
    param([string]$Query)
    $url = "$AdvisorUrl/v1/chat/completions"
    $payload = @{
        model    = "oss"
        messages = @(@{ role = "user"; content = $Query })
        temperature = 0.2
        max_tokens  = 512
    } | ConvertTo-Json -Depth 6 -Compress
    try {
        $resp = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 120
        $text = $resp.choices[0].message.content
        return [ordered]@{ ok = $true; status = "advisor"; source = "advisor"; text = $text; governing_track = "Advisor" }
    } catch {
        return [ordered]@{ ok = $false; status = "advisor_offline"; source = "advisor"; text = "Advisor not available."; error = $_.Exception.Message }
    }
}

function Invoke-AdvisorEvolveQuery {
    param(
        [string]$MicronautId,
        [string]$MicronautDescription,
        [string]$UserQuery,
        [string[]]$SeedCapabilities,
        [string[]]$SeedGoverns
    )
    $url = "$AdvisorUrl/v1/chat/completions"
    $seedCapsJson = if ($SeedCapabilities.Count -gt 0) { $SeedCapabilities | ConvertTo-Json -Compress } else { '[]' }
    $seedGovsJson = if ($SeedGoverns.Count -gt 0)      { $SeedGoverns      | ConvertTo-Json -Compress } else { '[]' }
    $sysPrompt = @"
You are the KHANARY capability validator for micronaut "$MicronautId".
A user query produced a miss. You must:
1. Accept seed capabilities that are semantically correct for this micronaut.
2. Reject synthetic/internal caps (e.g. miss::*, needs_fold:*, research_pressure:*).
3. Rewrite vague caps into precise semantic names (verb:noun form, lowercase, no spaces, max 6 total).
4. Add implied capabilities the seed missed.
5. Write a short 1-2 sentence reply explaining what was learned.
Respond ONLY with valid JSON - no markdown fences:
{"reply":"...","accepted_capabilities":[...],"accepted_governs":[...]}
"@
    $userPrompt = "Micronaut: $MicronautId`nDescription: $MicronautDescription`nUser query: $UserQuery`nSeed capabilities: $seedCapsJson`nSeed governs: $seedGovsJson"
    $payload = @{
        model    = "oss"
        messages = @(
            @{ role = "system"; content = $sysPrompt }
            @{ role = "user";   content = $userPrompt }
        )
        temperature = 0.15
        max_tokens  = 300
    } | ConvertTo-Json -Depth 8 -Compress
    try {
        $resp = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 120
        $text = $resp.choices[0].message.content
        $stripped = $text -replace '(?s)^```[a-z]*\s*', '' -replace '(?s)\s*```$', ''
        try {
            $delta = $stripped | ConvertFrom-Json
            return [ordered]@{
                ok                    = $true
                reply                 = if ($delta.reply) { [string]$delta.reply } else { "Learned from this query." }
                accepted_capabilities = if ($delta.accepted_capabilities) { @($delta.accepted_capabilities) } else { @() }
                accepted_governs      = if ($delta.accepted_governs)      { @($delta.accepted_governs) }      else { @() }
            }
        } catch {
            return [ordered]@{ ok = $false; reply = $text; accepted_capabilities = @(); accepted_governs = @() }
        }
    } catch {
        return [ordered]@{ ok = $false; reply = "Advisor not available."; accepted_capabilities = @(); accepted_governs = @() }
    }
}

function Invoke-LfmUnfoldSummary {
    # Receives the fold_graph_parse sub-object {atoms, candidates, unresolved, ...}.
    # LFM may only narrate what F_Q supplies: known <= Resolved(F_Q), uncertain <= LowEvidence(F_Q),
    # unresolved <= Unresolved(F_Q). LFM may not enlarge F_Q.
    param(
        [string]$Query,
        [object]$FoldParse,   # fold_graph_parse sub-object, NOT the full RPC result
        [string]$BaseUrl
    )
    $knownAtoms      = @()
    $uncertainAtoms  = @()
    $unresolvedToks  = @()
    try {
        foreach ($c in $FoldParse.candidates) {
            if ([int]$c.evidence_count -ge 2) { $knownAtoms    += [string]$c.atom }
            else                               { $uncertainAtoms += [string]$c.atom }
        }
    } catch {}
    try { $unresolvedToks = @($FoldParse.unresolved | ForEach-Object { [string]$_.token } | Where-Object { $_ }) } catch {}

    $knownStr      = if ($knownAtoms.Count -gt 0)     { $knownAtoms -join ', ' }     else { "(none resolved)" }
    $uncertainStr  = if ($uncertainAtoms.Count -gt 0) { $uncertainAtoms -join ', ' } else { "(none)" }
    $unresolvedStr = if ($unresolvedToks.Count -gt 0)  { $unresolvedToks -join ', ' } else { "(none)" }

    $sysPrompt = @"
You are a knowledge-state reporter for KHANARY. Project the Fold Graph data into language.
STRICT RULES - your output must satisfy these set constraints:
  known items        MUST come only from the resolved-atoms list.
  uncertain items    MUST come only from the low-evidence-atoms list.
  unresolved items   MUST come only from the unresolved-tokens list.
  update_candidates  are ADVISORY ONLY - do not issue mutation instructions.
  Do NOT add any atom, concept, or fact not present in the data below.
Respond ONLY with valid JSON (no markdown fences):
{"summary":"...","known":[],"uncertain":[],"unresolved":[],"update_candidates":[],"question":"..."}
"@
    $userPrompt = "Query: $Query`nResolved atoms (evidence_count>=2, known): $knownStr`nLow-evidence atoms (evidence_count=1, uncertain): $uncertainStr`nUnresolved tokens (absent from semantic field): $unresolvedStr`n`nFormulate exactly one question whose answer would reduce the principal uncertainty."

    $url = "$BaseUrl/v1/chat/completions"
    $payload = @{
        model    = "LFM2.5-1.2B-Instruct-Q8_0"
        messages = @(
            @{ role = "system"; content = $sysPrompt }
            @{ role = "user";   content = $userPrompt }
        )
        temperature = 0.1
        max_tokens  = 128
    } | ConvertTo-Json -Depth 8 -Compress
    try {
        $resp    = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 60
        $text    = $resp.choices[0].message.content
        if ([string]::IsNullOrWhiteSpace($text)) { return $null }
        $stripped = $text -replace '(?s)^```[a-z]*\s*', '' -replace '(?s)\s*```$', ''
        try {
            $p = $stripped | ConvertFrom-Json
            return [ordered]@{
                ok                = $true
                summary           = if ($p.summary)           { [string]$p.summary }        else { $stripped.Trim() }
                known             = if ($p.known)             { @($p.known) }               else { $knownAtoms }
                uncertain         = if ($p.uncertain)         { @($p.uncertain) }           else { $uncertainAtoms }
                unresolved        = if ($p.unresolved)        { @($p.unresolved) }          else { $unresolvedToks }
                update_candidates = if ($p.update_candidates) { @($p.update_candidates) }  else { @() }
                question          = if ($p.question)          { [string]$p.question }       else { "" }
            }
        } catch {
            return [ordered]@{
                ok = $true; summary = $stripped.Trim(); known = $knownAtoms
                uncertain = $uncertainAtoms; unresolved = $unresolvedToks
                update_candidates = @(); question = ""
            }
        }
    } catch {
        return $null
    }
}

function Invoke-TrackMuQuery {
    param(
        [string]$Query,
        [string]$BaseUrl,
        [string]$ModelId,
        [string]$TrackLabel,
        [double]$Temperature = 0.3,
        [int]$MaxTokens = 64
    )
    $url = "$BaseUrl/v1/chat/completions"
    $payload = @{
        model    = $ModelId
        messages = @(@{ role = "user"; content = $Query })
        temperature = $Temperature
        max_tokens  = $MaxTokens
    } | ConvertTo-Json -Depth 6 -Compress
    try {
        $resp = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 180
        $content = $resp.choices[0].message.content
        if ([string]::IsNullOrWhiteSpace($content)) { return $null }
        $keySlug = ($TrackLabel.ToLowerInvariant() -replace '[^\x00-\x7F]', '' -replace '[^a-z0-9]', '_').TrimEnd('_')
        try {
            Invoke-JsonRpcCall -Url $RpcUrl -MethodName "record_key" -ParamsObject @{
                key      = "${keySlug}:$Query"
                response = $content
            } | Out-Null
        } catch {}
        return [ordered]@{ ok = $true; status = "mu_response"; source = $TrackLabel; text = $content; governing_track = $TrackLabel }
    } catch {
        return $null
    }
}

switch ($Action) {
    "aiml" {
        $inputText = if ([string]::IsNullOrWhiteSpace($Query)) { "status" } else { $Query.Trim() }
        $skipRouteMap = $inputText -match '(?i)\b(timer|countdown)\b'
        $route = $null
        if (-not $skipRouteMap) {
            $route = Resolve-AimlToolRoute -InputText $inputText
        }
        if ($null -ne $route) {
            $surface = [string]$route["surface"]
            $methodName = [string]$route["method"]
            if ([string]::IsNullOrWhiteSpace($surface) -or [string]::IsNullOrWhiteSpace($methodName)) {
                throw "Invalid tool route row in ${toolVerbMapFile}: missing surface/method."
            }
            $paramsObj = @{}
            if ($route.ContainsKey("params") -and $null -ne $route["params"]) {
                $paramsObj = @{} + $route["params"]
            }
            if ($paramsObj.ContainsKey("query") -and [string]$paramsObj["query"] -eq "__USER_QUERY__") {
                $paramsObj["query"] = $inputText
            }
            $targetUrl = if ($surface -eq "mcp") { $McpUrl } else { $RpcUrl }
            $result = Invoke-JsonRpcCall -Url $targetUrl -MethodName $methodName -ParamsObject $paramsObj
            Emit-Response $result
            break
        }

        $preferAimlCatalog = $inputText -match '(?i)^\s*tool\s+key(s)?\b'
        $preferDirectResearch =
            $inputText -match '(?i)\bsearch( the)? web\b' -or
            $inputText -match '(?i)\bweb research\b' -or
            $inputText -match '(?i)\brecent news\b' -or
            $inputText -match '(?i)\blatest news\b' -or
            $inputText -match '(?i)\bheadlines?\b'
        if ($inputText -match '(?i)\bask\s+my\s+advisor\b' -or
            $inputText -match '(?i)\bi\s+(don''?t|do\s+not)\s+know\s+the\s+answer\b') {
            $preferDirectResearch = $false
        }
        $effective = $inputText
        if (-not $preferAimlCatalog -and -not $preferDirectResearch -and $effective -notmatch '^(Mx2LM:|@mx2lm)') {
            $effective = "Mx2LM: $effective"
        }
        try {
            $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "key_query" -ParamsObject @{ query = $effective; aiml_file = $aimlFile; route_map = $toolVerbMapFile }
            if ($preferDirectResearch) {
                $planPayload = $result
                if ($null -ne $result -and $null -ne $result.PSObject -and $result.PSObject.Properties.Match("result").Count -gt 0) {
                    $planPayload = $result.result
                }
                $source = Read-ObjectString -Object $planPayload -Name "source"
                $status = Read-ObjectString -Object $planPayload -Name "status"
                $isResearchDispatch =
                    $source -ieq "web-research" -and
                    ($status -ieq "research_plan_ready" -or $status -ieq "web_research_dispatched")
                if ($isResearchDispatch) {
                    $researchResponse = Build-WebResearchResponse -InputText $inputText -PlanPayload $planPayload
                    Emit-ResultPayload -ResultPayload $researchResponse
                    break
                }
            }
            # µ-model routing and miss->advisor fallback
            $rInner    = if ($null -ne $result -and $null -ne $result.PSObject -and $result.PSObject.Properties.Match("result").Count -gt 0) { $result.result } else { $result }
            $rPhase    = Read-ObjectString -Object $rInner -Name "phase"
            $rMissKind = Read-ObjectString -Object $rInner -Name "miss_kind"
            $rStatus   = Read-ObjectString -Object $rInner -Name "status"
            $rGovTrack = Read-ObjectString -Object $rInner -Name "governing_track"
            if ([string]::IsNullOrWhiteSpace($rGovTrack)) {
                $rGovTrack = Read-ObjectString -Object $rInner -Name "track"
            }
            $rActionOps = @()
            if ($null -ne $rInner -and $null -ne $rInner.PSObject -and $rInner.PSObject.Properties.Match("action_grams").Count -gt 0) {
                foreach ($ag in @($rInner.action_grams)) {
                    if ($null -eq $ag) { continue }
                    $op = Read-ObjectString -Object $ag -Name "op"
                    if (-not [string]::IsNullOrWhiteSpace($op)) {
                        $rActionOps += $op
                    }
                }
            }

            $triggerAdvisor = ($rActionOps -contains "advisor_web_research_trigger") -or ($rGovTrack -match '(?i)\badvisor\b')
            if ($triggerAdvisor -and -not [string]::IsNullOrWhiteSpace($AdvisorUrl)) {
                try {
                    $advisorQuery = "$inputText`n`nContext: MISS or low-confidence semantic gap. Build a bounded web-research + advisor plan."
                    $advisorHit = Invoke-AdvisorQuery -Query $advisorQuery
                    if ($advisorHit.ok -and -not [string]::IsNullOrWhiteSpace($advisorHit.text)) {
                        Emit-ResultPayload -ResultPayload ([ordered]@{
                            ok              = $true
                            status          = "advisor_triggered"
                            source          = "advisor"
                            text            = $advisorHit.text
                            governing_track = "ADVISOR-µ"
                            trigger         = "advisor_web_research_trigger"
                        })
                        break
                    }
                } catch {}
            }

            $triggerLfmToolHelp = ($rActionOps -contains "lfm_tool_help_trigger")
            if ($triggerLfmToolHelp -and -not [string]::IsNullOrWhiteSpace($ChatMuUrl)) {
                try {
                    $lfmPrompt = "Tool-selection assist request: $inputText"
                    $muResult = Invoke-TrackMuQuery -Query $lfmPrompt -BaseUrl $ChatMuUrl -ModelId "LFM2.5-1.2B-Instruct-Q8_0" -TrackLabel "CHAT-µ" -Temperature 0.25 -MaxTokens 96
                    if ($null -ne $muResult) {
                        $muResult.status = "lfm_tool_help_triggered"
                        $muResult.trigger = "lfm_tool_help_trigger"
                        Emit-ResultPayload -ResultPayload $muResult
                        break
                    }
                } catch {}
            }

            if ($rGovTrack -match '(?i)\bchat\b' -and -not [string]::IsNullOrWhiteSpace($ChatMuUrl)) {
                $muResult = Invoke-TrackMuQuery -Query $inputText -BaseUrl $ChatMuUrl -ModelId "LFM2.5-1.2B-Instruct-Q8_0" -TrackLabel "CHAT-µ" -Temperature 0.3 -MaxTokens 64
                if ($null -ne $muResult) { Emit-ResultPayload -ResultPayload $muResult; break }
            }
            if ($rGovTrack -match '(?i)\btool\b' -and -not [string]::IsNullOrWhiteSpace($ToolMuUrl)) {
                $muResult = Invoke-TrackMuQuery -Query $inputText -BaseUrl $ToolMuUrl -ModelId "gemma-3-1b-it-qat-q4_0" -TrackLabel "TOOL-µ" -Temperature 0.1 -MaxTokens 64
                if ($null -ne $muResult) { Emit-ResultPayload -ResultPayload $muResult; break }
            }
            $isMiss = ($rPhase -match '(?i)fallback_projection') -or
                      (-not [string]::IsNullOrWhiteSpace($rMissKind)) -or
                      ($rStatus -ieq "miss")
            if ($isMiss -and -not [string]::IsNullOrWhiteSpace($AdvisorUrl)) {
                # Stage 1 - fold_graph_parse -> LFM projection (CHAT-µ narrates F_Q; does not enlarge it)
                $lfmResult      = $null
                $lfmSummaryText = $null
                if (-not [string]::IsNullOrWhiteSpace($ChatMuUrl)) {
                    try {
                        $foldRpc = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "fold_graph_parse" -ParamsObject @{ query = $inputText }
                        # fold_graph_parse is nested: result.fold_graph_parse
                        $fgpInner = $null
                        if ($foldRpc.result -and $foldRpc.result.fold_graph_parse)      { $fgpInner = $foldRpc.result.fold_graph_parse }
                        elseif ($foldRpc.fold_graph_parse)                               { $fgpInner = $foldRpc.fold_graph_parse }
                        if ($null -ne $fgpInner) {
                            $lfmResult = Invoke-LfmUnfoldSummary -Query $inputText -FoldParse $fgpInner -BaseUrl $ChatMuUrl
                            if ($null -ne $lfmResult -and $lfmResult.ok) { $lfmSummaryText = $lfmResult.summary }
                        }
                    } catch {}
                }

                # Stage 2 - Advisor: cap rewrite (pendingVerify) or free-text elaboration
                $advisorText   = $null
                $pendingVerify = $false
                try { $pendingVerify = [bool]$rInner.factory_mutation.pending_advisor_verify } catch {}
                if ($pendingVerify) {
                    $fm            = $rInner.factory_mutation
                    $micronautId   = if ($fm.micronaut)                 { [string]$fm.micronaut }                 else { "" }
                    $micronautDesc = if ($rInner.micronaut_description) { [string]$rInner.micronaut_description } else { "" }
                    $seedCaps      = if ($fm.added_capabilities)        { @($fm.added_capabilities) }            else { @() }
                    $seedGovs      = if ($fm.added_governs)             { @($fm.added_governs) }                 else { @() }
                    $delta = Invoke-AdvisorEvolveQuery `
                        -MicronautId          $micronautId `
                        -MicronautDescription $micronautDesc `
                        -UserQuery            $inputText `
                        -SeedCapabilities     $seedCaps `
                        -SeedGoverns          $seedGovs
                    if ($delta.ok -and ($delta.accepted_capabilities.Count -gt 0 -or $delta.accepted_governs.Count -gt 0)) {
                        try {
                            Invoke-JsonRpcCall -Url $RpcUrl -MethodName "factory_advisor_commit" -ParamsObject @{
                                micronaut_id          = $micronautId
                                accepted_capabilities = $delta.accepted_capabilities
                                accepted_governs      = $delta.accepted_governs
                            } | Out-Null
                        } catch {}
                    }
                    if (-not [string]::IsNullOrWhiteSpace($delta.reply)) { $advisorText = $delta.reply }
                } else {
                    # Thread Q_gap from LFM into advisor - advisor answers from the gap, not just the raw query
                    $advisorQuery = $inputText
                    if ($null -ne $lfmResult -and -not [string]::IsNullOrWhiteSpace($lfmResult.question)) {
                        $advisorQuery = "$inputText`n`nKnowledge gap (from semantic field): $($lfmResult.question)"
                    }
                    $adResult = Invoke-AdvisorQuery -Query $advisorQuery
                    if ($adResult.ok -and -not [string]::IsNullOrWhiteSpace($adResult.text)) { $advisorText = $adResult.text }
                }

                # Stage 3 - Combine: LFM gram context + advisor elaboration/learned text
                $parts = @()
                if (-not [string]::IsNullOrWhiteSpace($lfmSummaryText)) { $parts += $lfmSummaryText }
                if (-not [string]::IsNullOrWhiteSpace($advisorText))    { $parts += $advisorText }
                $replyText = if ($parts.Count -gt 0) { $parts -join "`n`n" } else { "Advisor not available." }

                Emit-ResultPayload -ResultPayload ([ordered]@{
                    ok              = $true
                    status          = "advisor"
                    source          = "advisor"
                    text            = $replyText
                    governing_track = "Advisor"
                })
                break
            }
            Emit-Response $result
        } catch {
            $initialRuntimeError = $_.Exception.Message
            if ($inputText -match '(?i)^\s*(hi|hello|hey)\b') {
                try {
                    Ensure-KhanaryRuntime
                    $retry = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "key_query" -ParamsObject @{ query = $effective; aiml_file = $aimlFile; route_map = $toolVerbMapFile }
                    Emit-Response $retry
                } catch {
                    $runtimeError = if ([string]::IsNullOrWhiteSpace($_.Exception.Message)) {
                        $initialRuntimeError
                    } else {
                        $_.Exception.Message
                    }
                    $fallback = [ordered]@{
                        ok = $true
                        status = "local_fallback"
                        source = "commands.aiml.local"
                        text = "Hello. KHANARY runtime is not ready yet, but the WPF shell is online."
                        runtime_error = $runtimeError
                    }
                    if ($RawJson) {
                        @{ jsonrpc = "2.0"; result = $fallback } | ConvertTo-Json -Depth 20
                    } else {
                        $fallback | ConvertTo-Json -Depth 20
                    }
                }
            } else {
                throw
            }
        }
    }
    "status" {
        $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "key_query" -ParamsObject @{ query = "STATUS:" }
        Emit-Response $result
    }
    "tools" {
        $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "key_query" -ParamsObject @{ query = "TOOLS:" }
        Emit-Response $result
    }
    "dispatch" {
        $effective = if ([string]::IsNullOrWhiteSpace($Query)) { "status" } else { $Query }
        $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "micronaut_dispatch" -ParamsObject @{ query = $effective }
        Emit-Response $result
    }
    "keyquery" {
        $effective = if ([string]::IsNullOrWhiteSpace($Query)) { "STATUS:" } else { $Query }
        $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "key_query" -ParamsObject @{ query = $effective }
        Emit-Response $result
    }
    "center" {
        $effective = if ([string]::IsNullOrWhiteSpace($Query)) { "STATUS:" } else { $Query }
        $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName "key_query" -ParamsObject @{ query = $effective }
        Emit-Response $result
    }
    "rpc" {
        if ([string]::IsNullOrWhiteSpace($Method)) {
            throw "rpc action requires -Method."
        }
        $paramsObj = Parse-ParamsJson -JsonText $ParamsJson
        $result = Invoke-JsonRpcCall -Url $RpcUrl -MethodName $Method -ParamsObject $paramsObj
        Emit-Response $result
    }
    "mcp" {
        if ([string]::IsNullOrWhiteSpace($Tool)) {
            throw "mcp action requires -Tool."
        }
        $paramsObj = Parse-ParamsJson -JsonText $ParamsJson
        $result = Invoke-JsonRpcCall -Url $McpUrl -MethodName $Tool -ParamsObject $paramsObj
        Emit-Response $result
    }
    "advisor_consult" {
        $q = if ([string]::IsNullOrWhiteSpace($Query)) { "What is your status?" } else { $Query }
        $advisorResult = Invoke-AdvisorQuery -Query $q
        Emit-ResultPayload -ResultPayload $advisorResult
    }
}
'@

Write-GeneratedFile -RelativePath "evolution\engine.runtime.json" -Content @"
{
  "schema": "mx2.evolution.runtime.v1",
  "bundle": "$BundleName",
  "generated_utc": "$utcNow",
  "strategy": "deterministic-path-based",
  "state_file": "evolution/state.json",
  "rules_file": "evolution/rules.xjson"
}
"@

Write-GeneratedFile -RelativePath "evolution\rules.xjson" -Content @"
{
  "@schema": "mx2.evolution.rules.v1",
  "rules": [
    {
      "id": "preserve-layout",
      "when": "always",
      "do": "verify required files exist"
    },
    {
      "id": "self-reference-only",
      "when": "runtime path rewrite",
      "do": "enforce relative paths under bundle root"
    },
    {
      "id": "wpf-shell-first",
      "when": "cli launch",
      "do": "start WPF shell and proxy actions to agents/commands.ps1"
    }
  ]
}
"@

Write-GeneratedFile -RelativePath "scripts\launch-wpf-cli.ps1" -Content @'
[CmdletBinding()]
param(
    [string]$Configuration = "Debug"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectPath = Join-Path $bundleRoot "cli\wpf\KhanaryMicronautCli.Wpf.csproj"

if (-not (Test-Path $projectPath)) {
    throw "WPF project not found: $projectPath"
}

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw "dotnet SDK is required to run the WPF CLI."
}

& dotnet run --project $projectPath -c $Configuration
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\KhanaryMicronautCli.Wpf.csproj" -Content @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>WinExe</OutputType>
    <TargetFramework>net8.0-windows</TargetFramework>
    <UseWPF>true</UseWPF>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <AssemblyName>KhanaryMicronautCli.Wpf</AssemblyName>
    <RootNamespace>KhanaryMicronautCli.Wpf</RootNamespace>
  </PropertyGroup>
</Project>
'@

Write-GeneratedFile -RelativePath "cli\wpf\App.xaml" -Content @'
<Application x:Class="KhanaryMicronautCli.Wpf.App"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             StartupUri="MainWindow.xaml">
    <Application.Resources />
</Application>
'@

Write-GeneratedFile -RelativePath "cli\wpf\App.xaml.cs" -Content @'
using System.Windows;

namespace KhanaryMicronautCli.Wpf;

public partial class App : Application
{
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\Threading.cs" -Content @'
using System;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace KhanaryMicronautCli.Wpf;

public static class Threading
{
    public static Task RunOnUiAsync(Dispatcher dispatcher, Action action)
    {
        if (dispatcher.CheckAccess())
        {
            action();
            return Task.CompletedTask;
        }
        return dispatcher.InvokeAsync(action).Task;
    }

    public static async void FireAndForget(Task task, Action<Exception>? onError = null)
    {
        try
        {
            await task.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            onError?.Invoke(ex);
        }
    }
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\Json.cs" -Content @'
using System;
using System.IO;
using System.Text.Json;

namespace KhanaryMicronautCli.Wpf;

public static class Json
{
    public static string Pretty(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        try
        {
            using var doc = JsonDocument.Parse(value);
            return JsonSerializer.Serialize(doc.RootElement, new JsonSerializerOptions
            {
                WriteIndented = true
            });
        }
        catch
        {
            return value;
        }
    }

    public static string FirstLine(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }
        using var reader = new StringReader(value.Trim());
        return reader.ReadLine() ?? string.Empty;
    }
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\Routes.cs" -Content @'
using System.Collections.Generic;

namespace KhanaryMicronautCli.Wpf;

public sealed record Route(string Action, string Description);

public static class Routes
{
    private static readonly Dictionary<string, Route> Map = new()
    {
        ["aiml"] = new Route("aiml", "AIML ngram tool-router"),
        ["keyquery"] = new Route("keyquery", "Direct key_query lane"),
        ["dispatch"] = new Route("dispatch", "Micronaut dispatch route"),
        ["status"] = new Route("status", "Runtime status snapshot"),
        ["tools"] = new Route("tools", "Available tools snapshot"),
        ["center"] = new Route("center", "Command-center style route")
    };

    public static Route Resolve(string action)
    {
        if (Map.TryGetValue(action, out var route))
        {
            return route;
        }
        return new Route("keyquery", "Fallback key_query route");
    }

    public static string CommandPreview(string action)
    {
        var route = Resolve(action);
        return $"{PowerShellProcess.CommandPreviewShell} -NoProfile -ExecutionPolicy Bypass -File agents\\commands.ps1 -Action {route.Action} -QueryBase64 <base64>";
    }
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\Ngrams.cs" -Content @'
using System;
using System.Collections.Generic;
using System.Linq;

namespace KhanaryMicronautCli.Wpf;

public static class Ngrams
{
    // Customization surface:
    // - Edit IntentAliases to rename concepts in the UI.
    // - Edit ActionAliases to rename deterministic action ops in the UI.
    // - Edit NormalizeNgram to change ngram display cleanup behavior.
    private static readonly Dictionary<string, string> IntentAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["intent:salutation"] = "salutation",
        ["intent:wellbeing_check"] = "wellbeing_check",
        ["intent:identity_query"] = "identity_query",
        ["intent:capability_query"] = "capability_query",
        ["intent:definition_query"] = "definition_query",
        ["intent:math_capability_query"] = "math_capability_query",
        ["phase:pop.neutral"] = "phase_neutral",
        ["phase:wo.random_chat_context"] = "phase_random_chat",
        ["phase:yax.learned_slang"] = "phase_learned_slang",
        ["phase:yax.learned_wave_slang"] = "phase_wave_slang",
        ["phase:sek.locale_language"] = "phase_locale_language"
    };

    private static readonly Dictionary<string, string> ActionAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["salutation"] = "salutation",
        ["wellbeing_check"] = "wellbeing_check",
        ["identity_query"] = "identity_query",
        ["capability_query"] = "capability_query",
        ["definition_query"] = "definition_query",
        ["math_capability_query"] = "math_capability_query"
    };

    private static readonly Dictionary<string, string> FrameAccentHex = new(StringComparer.OrdinalIgnoreCase)
    {
        ["default"] = "#4EA1FF",
        ["salutation"] = "#63D3FF",
        ["wellbeing"] = "#6BE38A",
        ["research"] = "#C58BFF",
        ["math"] = "#FFB35C",
        ["timer"] = "#FFD966",
        ["clock"] = "#8FE0FF",
        ["definition"] = "#9BDE8E"
    };

    public static string ResolveSvgCFrame(
        string source,
        string status,
        IEnumerable<string> intents,
        IEnumerable<string> actions)
    {
        var src = (source ?? string.Empty).Trim().ToLowerInvariant();
        var st = (status ?? string.Empty).Trim().ToLowerInvariant();
        var joinedIntents = string.Join("|", intents.Select(NormalizeIntent)).ToLowerInvariant();
        var joinedActions = string.Join("|", actions.Select(NormalizeActionLabel)).ToLowerInvariant();

        if (src == "utility-runtime")
        {
            if (st == "timer_started" || st == "timer_status") return "timer";
            if (st.StartsWith("clock_")) return "clock";
            if (st == "calculation_complete") return "math";
            if (st == "knowledge_snapshot" || st == "fact_lookup") return "definition";
        }
        if (src == "web_research") return "research";
        if (joinedIntents.Contains("phase_wave_slang")) return "timer";
        if (joinedIntents.Contains("phase_learned_slang")) return "math";
        if (joinedIntents.Contains("phase_random_chat")) return "clock";
        if (joinedIntents.Contains("phase_locale_language")) return "salutation";
        if (joinedActions.Contains("math_capability_query")) return "math";
        if (joinedActions.Contains("definition_query") || joinedIntents.Contains("definition_query")) return "definition";
        if (joinedIntents.Contains("wellbeing_check")) return "wellbeing";
        if (joinedIntents.Contains("salutation")) return "salutation";
        return "default";
    }

    public static string ResolveFrameAccentHex(string frameKey)
    {
        var key = string.IsNullOrWhiteSpace(frameKey) ? "default" : frameKey.Trim();
        if (FrameAccentHex.TryGetValue(key, out var hex))
            return hex;
        return FrameAccentHex["default"];
    }

    public static string NormalizeNgram(string gram)
    {
        if (string.IsNullOrWhiteSpace(gram)) return string.Empty;
        return gram.Trim().ToLowerInvariant();
    }

    public static string NormalizeIntent(string intent)
    {
        if (string.IsNullOrWhiteSpace(intent)) return string.Empty;
        var key = intent.Trim();
        if (IntentAliases.TryGetValue(key, out var alias))
            return alias;
        return key;
    }

    public static string NormalizeActionLabel(string actionLabel)
    {
        if (string.IsNullOrWhiteSpace(actionLabel)) return string.Empty;
        var label = actionLabel.Trim();
        var arrowIdx = label.LastIndexOf("=>", StringComparison.Ordinal);
        if (arrowIdx <= 0 || arrowIdx + 2 >= label.Length)
            return label;
        var left = label.Substring(0, arrowIdx).Trim();
        var right = label.Substring(arrowIdx + 2).Trim();
        if (ActionAliases.TryGetValue(right, out var alias))
            right = alias;
        return $"{left} => {right}";
    }

    public static string JoinNgrams(IEnumerable<string> ngrams)
    {
        var items = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var gram in ngrams)
        {
            var normalized = NormalizeNgram(gram);
            if (!string.IsNullOrWhiteSpace(normalized))
                items.Add(normalized);
        }
        return items.Count == 0 ? "none" : string.Join(" | ", items.OrderBy(v => v, StringComparer.OrdinalIgnoreCase));
    }

    public static string JoinConcepts(IEnumerable<string> intents)
    {
        var items = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var intent in intents)
        {
            var normalized = NormalizeIntent(intent);
            if (!string.IsNullOrWhiteSpace(normalized))
                items.Add(normalized);
        }
        return items.Count == 0 ? string.Empty : string.Join(" | ", items.OrderBy(v => v, StringComparer.OrdinalIgnoreCase));
    }

    public static string JoinActions(IEnumerable<string> actions)
    {
        var items = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var action in actions)
        {
            var normalized = NormalizeActionLabel(action);
            if (!string.IsNullOrWhiteSpace(normalized))
                items.Add(normalized);
        }
        return items.Count == 0 ? string.Empty : string.Join(" | ", items.OrderBy(v => v, StringComparer.OrdinalIgnoreCase));
    }

    public static string JoinLearning(IEnumerable<string> learnedFacts)
    {
        var items = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var fact in learnedFacts)
        {
            var normalized = fact?.Trim() ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(normalized))
                items.Add(normalized);
        }
        return items.Count == 0 ? string.Empty : string.Join(" | ", items.OrderBy(v => v, StringComparer.OrdinalIgnoreCase));
    }
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\PowerShellProcess.cs" -Content @'
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace KhanaryMicronautCli.Wpf;

public static class PowerShellProcess
{
    private const int StartupTimeoutSeconds = 45;
    private const int CommandTimeoutSeconds = 90;

    public static string CommandPreviewShell =>
        Path.GetFileNameWithoutExtension(ShellExecutable).Equals("pwsh", StringComparison.OrdinalIgnoreCase)
            ? "pwsh"
            : "powershell";

    private static readonly string ShellExecutable = ResolveShellExecutable();

    private static string ResolveShellExecutable()
    {
        var pathValue = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (var segment in pathValue.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var folder = segment.Trim();
            if (folder.Length == 0)
            {
                continue;
            }

            var candidate = Path.Combine(folder, "pwsh.exe");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return "powershell.exe";
    }

    public static async Task<(string StdOut, string StdErr, int ExitCode)> RunAsync(
        string workingDirectory,
        string scriptPath,
        string action,
        string query)
    {
        var encoded = Convert.ToBase64String(Encoding.UTF8.GetBytes(query ?? string.Empty));
        var args =
            $"-NoProfile -ExecutionPolicy Bypass -File \"{scriptPath}\" -Action {action} -QueryBase64 {encoded} -StartupTimeoutSec {StartupTimeoutSeconds} -RawJson";

        var psi = new ProcessStartInfo(ShellExecutable, args)
        {
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = psi };
        process.Start();
        var outTask = process.StandardOutput.ReadToEndAsync();
        var errTask = process.StandardError.ReadToEndAsync();
        var waitForExitTask = process.WaitForExitAsync();
        var completedTask = await Task.WhenAny(waitForExitTask, Task.Delay(TimeSpan.FromSeconds(CommandTimeoutSeconds)));

        if (completedTask != waitForExitTask)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // best effort kill
            }

            await waitForExitTask;
            var timeoutOut = await outTask;
            var timeoutErr = await errTask;
            var timeoutMessage = $"Timed out waiting for runtime response after {CommandTimeoutSeconds}s.";
            if (string.IsNullOrWhiteSpace(timeoutErr))
            {
                timeoutErr = timeoutMessage;
            }
            else
            {
                timeoutErr = timeoutErr.TrimEnd() + Environment.NewLine + timeoutMessage;
            }
            return (timeoutOut, timeoutErr, 124);
        }

        return (await outTask, await errTask, process.ExitCode);
    }
}
'@

Write-GeneratedFile -RelativePath "cli\wpf\MainWindow.xaml" -Content @'
<Window x:Class="KhanaryMicronautCli.Wpf.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="MX2LM WPF CLI"
        Height="860"
        Width="1320"
        WindowStartupLocation="CenterScreen"
        Background="#101316"
        Foreground="#E6E6E6"
        FontFamily="Consolas">
    <Window.Resources>
        <SolidColorBrush x:Key="ComboBoxBgBrush" Color="#0D1116" />
        <SolidColorBrush x:Key="ComboBoxPopupBrush" Color="#0B0F14" />
        <SolidColorBrush x:Key="ComboBoxBorderBrush" Color="#3A4652" />
        <SolidColorBrush x:Key="ComboBoxTextBrush" Color="#DCE8F5" />
        <SolidColorBrush x:Key="ComboBoxArrowBrush" Color="#9ED3FF" />
        <SolidColorBrush x:Key="ComboBoxHoverBrush" Color="#1A3A52" />
        <SolidColorBrush x:Key="ComboBoxSelectedBrush" Color="#24567A" />
        <SolidColorBrush x:Key="ComboBoxDisabledBrush" Color="#6B7682" />
        <SolidColorBrush x:Key="SendButtonBgBrush" Color="#1E4E73" />
        <SolidColorBrush x:Key="SendButtonBorderBrush" Color="#3E7EAE" />
        <SolidColorBrush x:Key="SendButtonHoverBrush" Color="#28618F" />
        <SolidColorBrush x:Key="SendButtonPressedBrush" Color="#17405D" />
        <SolidColorBrush x:Key="ClearButtonBgBrush" Color="#4A2A32" />
        <SolidColorBrush x:Key="ClearButtonBorderBrush" Color="#8D5A66" />
        <SolidColorBrush x:Key="ClearButtonHoverBrush" Color="#5F3641" />
        <SolidColorBrush x:Key="ClearButtonPressedBrush" Color="#3C2229" />
        <SolidColorBrush x:Key="ButtonTextBrush" Color="#F4FAFF" />
        <SolidColorBrush x:Key="ButtonDisabledBgBrush" Color="#222A31" />
        <SolidColorBrush x:Key="ButtonDisabledBorderBrush" Color="#343F49" />
        <SolidColorBrush x:Key="ButtonDisabledTextBrush" Color="#7B8794" />

        <Style x:Key="ManualColorComboBoxItemStyle" TargetType="{x:Type ComboBoxItem}">
            <Setter Property="Background" Value="{StaticResource ComboBoxPopupBrush}" />
            <Setter Property="Foreground" Value="{StaticResource ComboBoxTextBrush}" />
            <Setter Property="FontWeight" Value="SemiBold" />
            <Setter Property="Padding" Value="8,4" />
            <Setter Property="HorizontalContentAlignment" Value="Stretch" />
            <Setter Property="VerticalContentAlignment" Value="Center" />
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type ComboBoxItem}">
                        <Border x:Name="ItemBorder" Background="{TemplateBinding Background}" Padding="{TemplateBinding Padding}">
                            <ContentPresenter HorizontalAlignment="{TemplateBinding HorizontalContentAlignment}"
                                              VerticalAlignment="{TemplateBinding VerticalContentAlignment}" />
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsHighlighted" Value="True">
                                <Setter TargetName="ItemBorder" Property="Background" Value="{StaticResource ComboBoxHoverBrush}" />
                                <Setter Property="Foreground" Value="#FFFFFF" />
                            </Trigger>
                            <Trigger Property="IsSelected" Value="True">
                                <Setter TargetName="ItemBorder" Property="Background" Value="{StaticResource ComboBoxSelectedBrush}" />
                                <Setter Property="Foreground" Value="#FFFFFF" />
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter Property="Foreground" Value="{StaticResource ComboBoxDisabledBrush}" />
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>

        <Style x:Key="ManualColorComboBoxStyle" TargetType="{x:Type ComboBox}">
            <Setter Property="Foreground" Value="{StaticResource ComboBoxTextBrush}" />
            <Setter Property="Background" Value="{StaticResource ComboBoxBgBrush}" />
            <Setter Property="BorderBrush" Value="{StaticResource ComboBoxBorderBrush}" />
            <Setter Property="BorderThickness" Value="1" />
            <Setter Property="Padding" Value="6,2" />
            <Setter Property="FontSize" Value="15" />
            <Setter Property="FontWeight" Value="SemiBold" />
            <Setter Property="ItemContainerStyle" Value="{StaticResource ManualColorComboBoxItemStyle}" />
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type ComboBox}">
                        <Grid>
                            <Border x:Name="OuterBorder"
                                    Background="{TemplateBinding Background}"
                                    BorderBrush="{TemplateBinding BorderBrush}"
                                    BorderThickness="{TemplateBinding BorderThickness}"
                                    CornerRadius="2" />
                            <Grid>
                                <ContentPresenter x:Name="SelectionContent"
                                                  Margin="10,0,28,0"
                                                  VerticalAlignment="Center"
                                                  HorizontalAlignment="Left"
                                                  RecognizesAccessKey="True"
                                                  Content="{TemplateBinding SelectionBoxItem}"
                                                  ContentTemplate="{TemplateBinding SelectionBoxItemTemplate}"
                                                  ContentTemplateSelector="{TemplateBinding ItemTemplateSelector}" />
                                <Path x:Name="Arrow"
                                      Width="10"
                                      Height="6"
                                      HorizontalAlignment="Right"
                                      VerticalAlignment="Center"
                                      Margin="0,0,10,0"
                                      Data="M 0 0 L 5 6 L 10 0 Z"
                                      Fill="{StaticResource ComboBoxArrowBrush}" />
                            </Grid>
                            <Popup x:Name="Popup"
                                   Placement="Bottom"
                                   IsOpen="{TemplateBinding IsDropDownOpen}"
                                   Focusable="False"
                                   AllowsTransparency="True"
                                   PopupAnimation="Fade">
                                <Border x:Name="DropDownBorder"
                                        Margin="0,2,0,0"
                                        MinWidth="{TemplateBinding ActualWidth}"
                                        Background="{StaticResource ComboBoxPopupBrush}"
                                        BorderBrush="{StaticResource ComboBoxBorderBrush}"
                                        BorderThickness="1"
                                        CornerRadius="2">
                                    <ScrollViewer SnapsToDevicePixels="True" CanContentScroll="True">
                                        <StackPanel IsItemsHost="True" KeyboardNavigation.DirectionalNavigation="Contained" />
                                    </ScrollViewer>
                                </Border>
                            </Popup>
                        </Grid>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter TargetName="OuterBorder" Property="BorderBrush" Value="{StaticResource ComboBoxHoverBrush}" />
                            </Trigger>
                            <Trigger Property="IsDropDownOpen" Value="True">
                                <Setter TargetName="OuterBorder" Property="BorderBrush" Value="{StaticResource ComboBoxSelectedBrush}" />
                                <Setter TargetName="OuterBorder" Property="Background" Value="{StaticResource ComboBoxPopupBrush}" />
                            </Trigger>
                            <Trigger Property="IsKeyboardFocusWithin" Value="True">
                                <Setter TargetName="OuterBorder" Property="BorderBrush" Value="{StaticResource ComboBoxSelectedBrush}" />
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter TargetName="OuterBorder" Property="Background" Value="#1A1F25" />
                                <Setter Property="Foreground" Value="{StaticResource ComboBoxDisabledBrush}" />
                                <Setter TargetName="Arrow" Property="Fill" Value="{StaticResource ComboBoxDisabledBrush}" />
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>

        <Style x:Key="ManualSendButtonStyle" TargetType="{x:Type Button}">
            <Setter Property="Foreground" Value="{StaticResource ButtonTextBrush}" />
            <Setter Property="Background" Value="{StaticResource SendButtonBgBrush}" />
            <Setter Property="BorderBrush" Value="{StaticResource SendButtonBorderBrush}" />
            <Setter Property="BorderThickness" Value="1" />
            <Setter Property="Padding" Value="12,6" />
            <Setter Property="FontSize" Value="15" />
            <Setter Property="FontWeight" Value="Bold" />
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type Button}">
                        <Border x:Name="ButtonBorder"
                                Background="{TemplateBinding Background}"
                                BorderBrush="{TemplateBinding BorderBrush}"
                                BorderThickness="{TemplateBinding BorderThickness}"
                                CornerRadius="2"
                                Padding="{TemplateBinding Padding}">
                            <ContentPresenter HorizontalAlignment="Center"
                                              VerticalAlignment="Center"
                                              RecognizesAccessKey="True" />
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter TargetName="ButtonBorder" Property="Background" Value="{StaticResource SendButtonHoverBrush}" />
                            </Trigger>
                            <Trigger Property="IsPressed" Value="True">
                                <Setter TargetName="ButtonBorder" Property="Background" Value="{StaticResource SendButtonPressedBrush}" />
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter TargetName="ButtonBorder" Property="Background" Value="{StaticResource ButtonDisabledBgBrush}" />
                                <Setter TargetName="ButtonBorder" Property="BorderBrush" Value="{StaticResource ButtonDisabledBorderBrush}" />
                                <Setter Property="Foreground" Value="{StaticResource ButtonDisabledTextBrush}" />
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>

        <Style x:Key="ManualClearButtonStyle" TargetType="{x:Type Button}">
            <Setter Property="Foreground" Value="{StaticResource ButtonTextBrush}" />
            <Setter Property="Background" Value="{StaticResource ClearButtonBgBrush}" />
            <Setter Property="BorderBrush" Value="{StaticResource ClearButtonBorderBrush}" />
            <Setter Property="BorderThickness" Value="1" />
            <Setter Property="Padding" Value="12,6" />
            <Setter Property="FontSize" Value="15" />
            <Setter Property="FontWeight" Value="Bold" />
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type Button}">
                        <Border x:Name="ButtonBorder"
                                Background="{TemplateBinding Background}"
                                BorderBrush="{TemplateBinding BorderBrush}"
                                BorderThickness="{TemplateBinding BorderThickness}"
                                CornerRadius="2"
                                Padding="{TemplateBinding Padding}">
                            <ContentPresenter HorizontalAlignment="Center"
                                              VerticalAlignment="Center"
                                              RecognizesAccessKey="True" />
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter TargetName="ButtonBorder" Property="Background" Value="{StaticResource ClearButtonHoverBrush}" />
                            </Trigger>
                            <Trigger Property="IsPressed" Value="True">
                                <Setter TargetName="ButtonBorder" Property="Background" Value="{StaticResource ClearButtonPressedBrush}" />
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter TargetName="ButtonBorder" Property="Background" Value="{StaticResource ButtonDisabledBgBrush}" />
                                <Setter TargetName="ButtonBorder" Property="BorderBrush" Value="{StaticResource ButtonDisabledBorderBrush}" />
                                <Setter Property="Foreground" Value="{StaticResource ButtonDisabledTextBrush}" />
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
    </Window.Resources>
    <Grid Margin="12">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto" />
            <RowDefinition Height="*" />
            <RowDefinition Height="220" />
        </Grid.RowDefinitions>

        <Border Grid.Row="0" BorderBrush="#2F373D" BorderThickness="1" Padding="8" CornerRadius="4" Margin="0,0,0,8">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="Auto" />
                    <ColumnDefinition Width="220" />
                    <ColumnDefinition Width="12" />
                    <ColumnDefinition Width="*" />
                </Grid.ColumnDefinitions>
                <TextBlock Grid.Column="0" Text="MODE" VerticalAlignment="Center" Margin="0,0,10,0" Foreground="#8FA1B3" />
                <ComboBox Grid.Column="1"
                          x:Name="ModeComboBox"
                          SelectionChanged="ModeComboBox_OnSelectionChanged"
                          Style="{StaticResource ManualColorComboBoxStyle}">
                    <ComboBoxItem Content="AIML (Mx2LM)" Tag="aiml" />
                    <ComboBoxItem Content="KEYQUERY" Tag="keyquery" />
                    <ComboBoxItem Content="DISPATCH" Tag="dispatch" />
                    <ComboBoxItem Content="STATUS" Tag="status" />
                    <ComboBoxItem Content="TOOLS" Tag="tools" />
                    <ComboBoxItem Content="CENTER" Tag="center" />
                </ComboBox>
                <TextBox Grid.Column="3"
                         x:Name="CommandPreviewTextBox"
                         IsReadOnly="True"
                         TextWrapping="Wrap"
                         Background="#0D1116"
                         Foreground="#9ED3FF"
                         BorderBrush="#3A4652"
                         Margin="0"
                         Padding="8,6" />
            </Grid>
        </Border>

        <Grid Grid.Row="1" Margin="0,0,0,8">
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="2*" />
                <ColumnDefinition Width="8" />
                <ColumnDefinition Width="*" />
            </Grid.ColumnDefinitions>

            <Border Grid.Column="0" BorderBrush="#2F373D" BorderThickness="1" CornerRadius="4" Padding="8">
                <DockPanel>
                    <TextBlock DockPanel.Dock="Top"
                               Text="TERMINAL TRANSCRIPT"
                               Margin="0,0,0,6"
                               Foreground="#9FB3C8" />
                    <TextBox x:Name="TranscriptTextBox"
                             IsReadOnly="True"
                             AcceptsReturn="True"
                             VerticalScrollBarVisibility="Auto"
                             HorizontalScrollBarVisibility="Auto"
                             TextWrapping="Wrap"
                             Background="#0A0E12"
                             Foreground="#D7E2EE"
                             FontSize="16"
                             FontWeight="SemiBold"
                             BorderThickness="0" />
                </DockPanel>
            </Border>

            <Border Grid.Column="2" BorderBrush="#2F373D" BorderThickness="1" CornerRadius="4" Padding="8">
                <DockPanel>
                    <TextBlock DockPanel.Dock="Top"
                               Text="BOT RENDER OUTPUT"
                               Margin="0,0,0,6"
                               Foreground="#9FB3C8" />
                    <ScrollViewer x:Name="BotCardsScrollViewer"
                                  DockPanel.Dock="Top"
                                  Height="300"
                                  VerticalScrollBarVisibility="Auto"
                                  HorizontalScrollBarVisibility="Disabled"
                                  Margin="0,0,0,6"
                                  Background="#070B10"
                                  BorderBrush="#27303A"
                                  BorderThickness="1"
                                  Padding="6">
                        <StackPanel x:Name="BotCardsPanel" />
                    </ScrollViewer>
                    <TextBlock DockPanel.Dock="Top"
                               Text="STRUCTURED SNAPSHOT"
                               Margin="0,0,0,6"
                               Foreground="#7FA0BE" />
                    <TextBox x:Name="RawTextBox"
                             IsReadOnly="True"
                             AcceptsReturn="True"
                             VerticalScrollBarVisibility="Auto"
                             HorizontalScrollBarVisibility="Auto"
                             TextWrapping="Wrap"
                             Background="#05080B"
                             Foreground="#8FD18B"
                             FontSize="16"
                             FontWeight="SemiBold"
                             BorderThickness="0" />
                </DockPanel>
            </Border>
        </Grid>

        <Border Grid.Row="2" BorderBrush="#2F373D" BorderThickness="1" CornerRadius="4" Padding="8">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto" />
                    <RowDefinition Height="*" />
                    <RowDefinition Height="Auto" />
                </Grid.RowDefinitions>

                <TextBlock Grid.Row="0" Text="INPUT (Ctrl+Enter sends, Enter inserts newline)" Foreground="#9FB3C8" Margin="0,0,0,6" />

                <TextBox Grid.Row="1"
                         x:Name="InputTextBox"
                         AcceptsReturn="True"
                         VerticalScrollBarVisibility="Auto"
                         HorizontalScrollBarVisibility="Auto"
                         TextWrapping="Wrap"
                         PreviewKeyDown="InputTextBox_OnPreviewKeyDown"
                         Background="#0D1116"
                         Foreground="#FFFFFF"
                         BorderBrush="#3A4652"
                         FontSize="18"
                         FontWeight="Bold"
                         Padding="8"
                         MinHeight="130" />

                <StackPanel Grid.Row="2" Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,8,0,0">
                    <Button x:Name="SendButton"
                            Content="Send"
                            Width="110"
                            Margin="0,0,8,0"
                            Style="{StaticResource ManualSendButtonStyle}"
                            Click="SendButton_OnClick" />
                    <Button x:Name="ClearButton"
                            Content="Clear"
                            Width="110"
                            Style="{StaticResource ManualClearButtonStyle}"
                            Click="ClearButton_OnClick" />
                </StackPanel>
            </Grid>
        </Border>
    </Grid>
</Window>
'@

Write-GeneratedFile -RelativePath "cli\wpf\MainWindow.xaml.cs" -Content @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using IOPath = System.IO.Path;
using ShapePath = System.Windows.Shapes.Path;

namespace KhanaryMicronautCli.Wpf;

public partial class MainWindow : Window
{
    private readonly string _bundleRoot;
    private readonly string _commandShim;
    private bool _isRunning;
    private DispatcherTimer? _countdownTimer;
    private DateTimeOffset? _countdownEndsAt;
    private string _countdownTimerId = string.Empty;
    private string _lastRenderedOutput = string.Empty;
    private int _lastRenderedCountdownSeconds = -1;
    private const int MaxBotCards = 120;
    private DispatcherTimer? _avatarAnimationTimer;
    private readonly List<AnimatedSpriteState> _animatedSprites = new();
    private SpriteAtlas? _spriteAtlas;

    private sealed class AnimatedSpriteState
    {
        public Image Image { get; init; } = null!;
        public IReadOnlyList<ImageSource> FrameSources { get; init; } = Array.Empty<ImageSource>();
        public int Index { get; set; }
    }

    private sealed class SpriteAtlas
    {
        public SpriteAtlas(IReadOnlyDictionary<string, IReadOnlyList<ImageSource>> sequences)
        {
            Sequences = sequences;
        }

        public IReadOnlyDictionary<string, IReadOnlyList<ImageSource>> Sequences { get; }
    }

    public MainWindow()
    {
        InitializeComponent();
        _bundleRoot = FindBundleRoot();
        _commandShim = IOPath.Combine(_bundleRoot, "agents", "commands.ps1");
        ModeComboBox.SelectedIndex = 0;
        RenderCommandPreview();
        AppendTranscript("[ready] MX2LM WPF shell online");
        AppendTranscript($"[bundle] {_bundleRoot}");
    }

    private static string FindBundleRoot()
    {
        var cursor = new DirectoryInfo(AppContext.BaseDirectory);
        while (cursor is not null)
        {
            if (File.Exists(IOPath.Combine(cursor.FullName, "model.runtime.json")))
            {
                return cursor.FullName;
            }
            cursor = cursor.Parent;
        }
        return AppContext.BaseDirectory;
    }

    private string CurrentAction()
    {
        return (ModeComboBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "keyquery";
    }

    private void AppendTranscript(string line)
    {
        TranscriptTextBox.AppendText(line + Environment.NewLine);
        TranscriptTextBox.ScrollToEnd();
    }

    private void RenderCommandPreview()
    {
        var action = CurrentAction();
        CommandPreviewTextBox.Text = Routes.CommandPreview(action);
    }

    private async Task<(string StdOut, string StdErr, int ExitCode)> RunCommandAsync(string action, string query)
    {
        return await PowerShellProcess.RunAsync(_bundleRoot, _commandShim, action, query);
    }

    private static string FormatCountdown(int totalSeconds)
    {
        if (totalSeconds < 0) totalSeconds = 0;
        var span = TimeSpan.FromSeconds(totalSeconds);
        var hours = (int)span.TotalHours;
        if (hours > 0)
            return $"{hours:D2}:{span.Minutes:D2}:{span.Seconds:D2}";
        return $"{span.Minutes:D2}:{span.Seconds:D2}";
    }

    private void SetRenderedOutput(string displayText)
    {
        _lastRenderedOutput = displayText ?? string.Empty;
        RefreshCountdownOverlay();
    }

    private void RefreshCountdownOverlay()
    {
        if (_countdownEndsAt.HasValue && !string.IsNullOrWhiteSpace(_countdownTimerId))
        {
            var remaining = (int)Math.Ceiling((_countdownEndsAt.Value - DateTimeOffset.UtcNow).TotalSeconds);
            if (remaining < 0) remaining = 0;
            _lastRenderedCountdownSeconds = remaining;
            var line = $"countdown: {_countdownTimerId} {FormatCountdown(remaining)}";
            RawTextBox.Text = string.IsNullOrWhiteSpace(_lastRenderedOutput)
                ? line
                : _lastRenderedOutput + Environment.NewLine + line;
            return;
        }

        _lastRenderedCountdownSeconds = -1;
        RawTextBox.Text = _lastRenderedOutput;
    }

    private void StopCountdown(bool announceCompletion = false)
    {
        if (_countdownTimer is not null)
            _countdownTimer.Stop();

        if (announceCompletion && !string.IsNullOrWhiteSpace(_countdownTimerId))
            AppendTranscript($"[timer] {_countdownTimerId} complete");

        _countdownEndsAt = null;
        _countdownTimerId = string.Empty;
        RefreshCountdownOverlay();
    }

    private void CountdownTimer_OnTick(object? sender, EventArgs e)
    {
        if (!_countdownEndsAt.HasValue)
        {
            StopCountdown();
            return;
        }

        var remaining = (int)Math.Ceiling((_countdownEndsAt.Value - DateTimeOffset.UtcNow).TotalSeconds);
        if (remaining <= 0)
        {
            StopCountdown(announceCompletion: true);
            return;
        }

        if (remaining != _lastRenderedCountdownSeconds)
            RefreshCountdownOverlay();
    }

    private void StartOrRefreshCountdown(string timerId, int remainingSeconds)
    {
        if (remainingSeconds <= 0 || string.IsNullOrWhiteSpace(timerId))
            return;

        var isNewTimer = !string.Equals(_countdownTimerId, timerId, StringComparison.OrdinalIgnoreCase) ||
                         !_countdownEndsAt.HasValue;

        _countdownTimerId = timerId.Trim();
        _countdownEndsAt = DateTimeOffset.UtcNow.AddSeconds(remainingSeconds);

        if (_countdownTimer is null)
        {
            _countdownTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            _countdownTimer.Tick += CountdownTimer_OnTick;
        }
        _countdownTimer.Start();
        RefreshCountdownOverlay();

        if (isNewTimer)
            AppendTranscript($"[timer] {_countdownTimerId} countdown {FormatCountdown(remainingSeconds)}");
    }

    private static Brush BrushFromHex(string hex, Color fallback)
    {
        try
        {
            var parsed = ColorConverter.ConvertFromString(hex);
            if (parsed is Color c) return new SolidColorBrush(c);
        }
        catch
        {
            // fallback below
        }
        return new SolidColorBrush(fallback);
    }

    private static Int32Rect ToPixelRect(BitmapSource sheet, double leftRatio, double topRatio, double widthRatio, double heightRatio)
    {
        if (sheet.PixelWidth < 2 || sheet.PixelHeight < 2)
            return Int32Rect.Empty;

        var x = (int)Math.Round(sheet.PixelWidth * leftRatio);
        var y = (int)Math.Round(sheet.PixelHeight * topRatio);
        var width = (int)Math.Round(sheet.PixelWidth * widthRatio);
        var height = (int)Math.Round(sheet.PixelHeight * heightRatio);

        x = Math.Clamp(x, 0, Math.Max(0, sheet.PixelWidth - 2));
        y = Math.Clamp(y, 0, Math.Max(0, sheet.PixelHeight - 2));
        width = Math.Clamp(width, 2, sheet.PixelWidth - x);
        height = Math.Clamp(height, 2, sheet.PixelHeight - y);
        return new Int32Rect(x, y, width, height);
    }

    private static Int32Rect CellRect(
        BitmapSource sheet,
        int col,
        int row,
        int insetLeft = 0,
        int insetTop = 0,
        int insetRight = 0,
        int insetBottom = 0)
    {
        if (sheet.PixelWidth < 12 || sheet.PixelHeight < 8)
            return Int32Rect.Empty;

        var cellWidth = Math.Max(2, sheet.PixelWidth / 6);
        var cellHeight = Math.Max(2, sheet.PixelHeight / 4);
        var x = (col * cellWidth) + insetLeft;
        var y = (row * cellHeight) + insetTop;
        var width = cellWidth - insetLeft - insetRight;
        var height = cellHeight - insetTop - insetBottom;

        x = Math.Clamp(x, 0, Math.Max(0, sheet.PixelWidth - 2));
        y = Math.Clamp(y, 0, Math.Max(0, sheet.PixelHeight - 2));
        width = Math.Clamp(width, 2, sheet.PixelWidth - x);
        height = Math.Clamp(height, 2, sheet.PixelHeight - y);
        return new Int32Rect(x, y, width, height);
    }

    private static IReadOnlyList<ImageSource> BuildSequence(BitmapSource sheet, params Int32Rect[] frames)
    {
        var sequence = new List<ImageSource>();
        foreach (var frame in frames)
        {
            if (frame.Width < 2 || frame.Height < 2) continue;
            var cropped = new CroppedBitmap(sheet, frame);
            cropped.Freeze();
            sequence.Add(cropped);
        }
        return sequence;
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<ImageSource>> BuildSpriteSequences(BitmapSource sheet)
    {
        // Position-locked atlas extraction. The sheet is authored as 6x4 logical cells.
        var topThumbs = CellRect(sheet, 2, 0, insetLeft: 24, insetTop: 10, insetRight: 24, insetBottom: 14);
        var topThink = CellRect(sheet, 3, 0, insetLeft: 20, insetTop: 10, insetRight: 20, insetBottom: 14);
        var topPoint = CellRect(sheet, 4, 0, insetLeft: 20, insetTop: 10, insetRight: 16, insetBottom: 14);
        var topWave = CellRect(sheet, 5, 0, insetLeft: 12, insetTop: 10, insetRight: 10, insetBottom: 14);

        var row2Tablet = CellRect(sheet, 2, 1, insetLeft: 18, insetTop: 12, insetRight: 18, insetBottom: 14);
        var row2Shades = CellRect(sheet, 3, 1, insetLeft: 18, insetTop: 12, insetRight: 18, insetBottom: 14);
        var row2Laptop = CellRect(sheet, 5, 1, insetLeft: 10, insetTop: 12, insetRight: 8, insetBottom: 12);

        var row3Cheer = CellRect(sheet, 2, 2, insetLeft: 16, insetTop: 10, insetRight: 16, insetBottom: 18);
        var row3Relax = CellRect(sheet, 3, 2, insetLeft: 12, insetTop: 10, insetRight: 12, insetBottom: 18);
        var defaultSequence = BuildSequence(sheet, topPoint);

        var map = new Dictionary<string, IReadOnlyList<ImageSource>>(StringComparer.OrdinalIgnoreCase)
        {
            ["default"] = defaultSequence,
            ["salutation"] = BuildSequence(sheet, topWave),
            ["wellbeing"] = BuildSequence(sheet, row3Cheer),
            ["research"] = BuildSequence(sheet, row2Tablet),
            ["math"] = BuildSequence(sheet, row2Shades),
            ["timer"] = BuildSequence(sheet, row2Laptop),
            ["clock"] = BuildSequence(sheet, row3Relax),
            ["definition"] = BuildSequence(sheet, topThink)
        };

        var fallback = defaultSequence.Count > 0 ? defaultSequence : BuildSequence(sheet, topThumbs);
        foreach (var key in new[] { "salutation", "wellbeing", "research", "math", "timer", "clock", "definition" })
        {
            if (!map.TryGetValue(key, out var sequence) || sequence.Count == 0)
                map[key] = fallback;
        }

        return map;
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<ImageSource>> BuildStaticSpriteSequences(ImageSource singleImage)
    {
        var sequence = new List<ImageSource> { singleImage };
        return new Dictionary<string, IReadOnlyList<ImageSource>>(StringComparer.OrdinalIgnoreCase)
        {
            ["default"] = sequence,
            ["salutation"] = sequence,
            ["wellbeing"] = sequence,
            ["research"] = sequence,
            ["math"] = sequence,
            ["timer"] = sequence,
            ["clock"] = sequence,
            ["definition"] = sequence
        };
    }

    private string ResolveSpriteSheetPath()
    {
        foreach (var spriteName in new[] { "asx-atomic-block.png", "asx-atomic-blocks.png", "vault-bot-boy.png" })
        {
            var bundledPath = IOPath.Combine(_bundleRoot, "cli", "wpf", "assets", spriteName);
            if (File.Exists(bundledPath))
                return bundledPath;
        }

        var cursor = new DirectoryInfo(_bundleRoot);
        while (cursor is not null)
        {
            foreach (var spriteName in new[] { "asx-atomic-block.png", "asx-atomic-blocks.png", "vault-bot-boy.png" })
            {
                var candidate = IOPath.Combine(cursor.FullName, spriteName);
                if (File.Exists(candidate))
                    return candidate;
            }
            cursor = cursor.Parent;
        }

        return string.Empty;
    }

    private void EnsureSpriteAtlasLoaded()
    {
        if (_spriteAtlas is not null)
            return;

        var spritePath = ResolveSpriteSheetPath();
        if (string.IsNullOrWhiteSpace(spritePath) || !File.Exists(spritePath))
            return;

        try
        {
            var spriteSheet = new BitmapImage();
            spriteSheet.BeginInit();
            spriteSheet.CacheOption = BitmapCacheOption.OnLoad;
            spriteSheet.UriSource = new Uri(spritePath, UriKind.Absolute);
            spriteSheet.EndInit();
            spriteSheet.Freeze();

            var fileName = IOPath.GetFileName(spritePath);
            var singleImageMode = string.Equals(fileName, "asx-atomic-block.png", StringComparison.OrdinalIgnoreCase);
            var map = singleImageMode ? BuildStaticSpriteSequences(spriteSheet) : BuildSpriteSequences(spriteSheet);
            _spriteAtlas = new SpriteAtlas(map);
        }
        catch
        {
            _spriteAtlas = null;
        }
    }

    private IReadOnlyList<ImageSource>? ResolveSpriteSequence(string frameKey)
    {
        EnsureSpriteAtlasLoaded();
        if (_spriteAtlas is null)
            return null;

        var key = string.IsNullOrWhiteSpace(frameKey) ? "default" : frameKey.Trim();
        if (_spriteAtlas.Sequences.TryGetValue(key, out var sequence) && sequence.Count > 0)
            return sequence;
        if (_spriteAtlas.Sequences.TryGetValue("default", out var fallback) && fallback.Count > 0)
            return fallback;
        return null;
    }

    private void EnsureAvatarAnimationTimer()
    {
        if (_avatarAnimationTimer is null)
        {
            _avatarAnimationTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(180)
            };
            _avatarAnimationTimer.Tick += AvatarAnimationTimer_OnTick;
        }

        if (!_avatarAnimationTimer.IsEnabled)
            _avatarAnimationTimer.Start();
    }

    private void AvatarAnimationTimer_OnTick(object? sender, EventArgs e)
    {
        if (_animatedSprites.Count == 0)
        {
            _avatarAnimationTimer?.Stop();
            return;
        }

        foreach (var sprite in _animatedSprites)
        {
            if (sprite.FrameSources.Count < 2) continue;
            sprite.Index = (sprite.Index + 1) % sprite.FrameSources.Count;
            sprite.Image.Source = sprite.FrameSources[sprite.Index];
        }
    }

    private FrameworkElement BuildVaultBotBadge(string frameKey, Brush accent, out AnimatedSpriteState? animatedSprite)
    {
        animatedSprite = null;

        var frame = string.IsNullOrWhiteSpace(frameKey) ? "default" : frameKey.Trim().ToLowerInvariant();
        var sequence = ResolveSpriteSequence(frame);
        if (sequence is null || sequence.Count == 0)
            return BuildSvgCFrameBadge(frame, accent);

        var avatar = new Image
        {
            Width = 44,
            Height = 44,
            Stretch = Stretch.Uniform,
            Source = sequence[0],
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            SnapsToDevicePixels = true
        };

        var avatarShell = new Border
        {
            Width = 54,
            Height = 54,
            CornerRadius = new CornerRadius(8),
            BorderBrush = accent,
            BorderThickness = new Thickness(1.6),
            Background = new SolidColorBrush(Color.FromRgb(10, 15, 22)),
            Padding = new Thickness(3),
            Child = avatar
        };

        var frameLabel = new TextBlock
        {
            Text = $"VAULT/{frame}",
            Foreground = accent,
            FontSize = 9,
            FontWeight = FontWeights.SemiBold,
            HorizontalAlignment = HorizontalAlignment.Center,
            Margin = new Thickness(0, 3, 0, 0)
        };

        var stack = new StackPanel { Width = 64 };
        stack.Children.Add(avatarShell);
        stack.Children.Add(frameLabel);

        if (sequence.Count > 1)
        {
            animatedSprite = new AnimatedSpriteState
            {
                Image = avatar,
                FrameSources = sequence
            };
        }

        return stack;
    }

    private static FrameworkElement BuildSvgCFrameBadge(string frameKey, Brush accent)
    {
        var avatarGrid = new Grid { Width = 62, Height = 62 };
        var shell = new Ellipse
        {
            Width = 58,
            Height = 58,
            Stroke = accent,
            StrokeThickness = 2.0,
            Fill = new SolidColorBrush(Color.FromRgb(16, 22, 30)),
            Margin = new Thickness(2)
        };
        avatarGrid.Children.Add(shell);

        var face = new Ellipse
        {
            Width = 46,
            Height = 46,
            Fill = new SolidColorBrush(Color.FromRgb(28, 38, 52)),
            Margin = new Thickness(8)
        };
        avatarGrid.Children.Add(face);

        var leftEye = new Ellipse
        {
            Width = 6,
            Height = 6,
            Fill = accent,
            HorizontalAlignment = HorizontalAlignment.Left,
            VerticalAlignment = VerticalAlignment.Top,
            Margin = new Thickness(20, 22, 0, 0)
        };
        var rightEye = new Ellipse
        {
            Width = 6,
            Height = 6,
            Fill = accent,
            HorizontalAlignment = HorizontalAlignment.Left,
            VerticalAlignment = VerticalAlignment.Top,
            Margin = new Thickness(36, 22, 0, 0)
        };
        avatarGrid.Children.Add(leftEye);
        avatarGrid.Children.Add(rightEye);

        var frame = string.IsNullOrWhiteSpace(frameKey) ? "default" : frameKey.Trim().ToLowerInvariant();
        var mouthPath = frame switch
        {
            "wellbeing" => "M 18 40 Q 31 49 44 40",
            "research" => "M 18 42 Q 31 35 44 42",
            "timer" => "M 18 41 L 44 41",
            "clock" => "M 18 41 L 44 41",
            "math" => "M 18 40 Q 31 43 44 40",
            "definition" => "M 18 41 Q 31 46 44 41",
            _ => "M 18 41 Q 31 45 44 41"
        };
        var mouth = new ShapePath
        {
            Data = Geometry.Parse(mouthPath),
            Stroke = accent,
            StrokeThickness = 1.8,
            Stretch = Stretch.None
        };
        avatarGrid.Children.Add(mouth);

        if (frame == "research")
        {
            avatarGrid.Children.Add(new Ellipse
            {
                Width = 12,
                Height = 12,
                Stroke = accent,
                StrokeThickness = 1.4,
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 8, 8, 0)
            });
            avatarGrid.Children.Add(new Rectangle
            {
                Width = 2,
                Height = 8,
                Fill = accent,
                RenderTransform = new RotateTransform(35),
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 17, 8, 0)
            });
        }
        else if (frame == "math")
        {
            avatarGrid.Children.Add(new TextBlock
            {
                Text = "∑",
                Foreground = accent,
                FontSize = 14,
                FontWeight = FontWeights.Bold,
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Bottom,
                Margin = new Thickness(0, 0, 8, 6)
            });
        }
        else if (frame == "timer" || frame == "clock")
        {
            avatarGrid.Children.Add(new Ellipse
            {
                Width = 14,
                Height = 14,
                Stroke = accent,
                StrokeThickness = 1.4,
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Bottom,
                Margin = new Thickness(0, 0, 8, 8)
            });
            avatarGrid.Children.Add(new Rectangle
            {
                Width = 1.6,
                Height = 5,
                Fill = accent,
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Bottom,
                Margin = new Thickness(0, 0, 14, 11)
            });
        }

        var frameLabel = new TextBlock
        {
            Text = $"SVG-C/{frame}",
            Foreground = accent,
            FontSize = 10,
            FontWeight = FontWeights.SemiBold,
            HorizontalAlignment = HorizontalAlignment.Center,
            Margin = new Thickness(0, 4, 0, 0)
        };

        var stack = new StackPanel { Width = 86 };
        stack.Children.Add(avatarGrid);
        stack.Children.Add(frameLabel);
        return stack;
    }

    private static Border BuildMetadataChip(string label, string value, Brush accent)
    {
        return new Border
        {
            CornerRadius = new CornerRadius(8),
            BorderBrush = accent,
            BorderThickness = new Thickness(1),
            Background = new SolidColorBrush(Color.FromRgb(20, 28, 38)),
            Margin = new Thickness(0, 0, 6, 6),
            Padding = new Thickness(8, 3, 8, 3),
            Child = new TextBlock
            {
                Text = $"{label}: {value}",
                Foreground = Brushes.WhiteSmoke,
                FontSize = 11,
                FontWeight = FontWeights.SemiBold,
                TextWrapping = TextWrapping.Wrap
            }
        };
    }

    private void AddBotCard(RenderedResponse rendered)
    {
        if (BotCardsPanel is null || rendered is null) return;

        var frameKey = string.IsNullOrWhiteSpace(rendered.FrameKey) ? "default" : rendered.FrameKey;
        var accent = BrushFromHex(Ngrams.ResolveFrameAccentHex(frameKey), Color.FromRgb(78, 161, 255));
        var avatarBadge = BuildVaultBotBadge(frameKey, accent, out var animatedSprite);

        var layout = new Grid();
        layout.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        layout.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

        var avatarHost = new Border
        {
            Margin = new Thickness(0, 0, 10, 0),
            Child = avatarBadge
        };
        Grid.SetColumn(avatarHost, 0);
        layout.Children.Add(avatarHost);

        var bubble = new Border
        {
            CornerRadius = new CornerRadius(10),
            Background = new SolidColorBrush(Color.FromRgb(12, 18, 26)),
            BorderBrush = accent,
            BorderThickness = new Thickness(1.4),
            Padding = new Thickness(10, 8, 10, 8)
        };
        Grid.SetColumn(bubble, 1);

        var bubbleStack = new StackPanel();
        bubbleStack.Children.Add(new TextBlock
        {
            Text = string.IsNullOrWhiteSpace(rendered.BotText) ? "(no text)" : rendered.BotText,
            Foreground = Brushes.White,
            FontSize = 15,
            FontWeight = FontWeights.SemiBold,
            TextWrapping = TextWrapping.Wrap
        });

        var chipWrap = new WrapPanel { Margin = new Thickness(0, 8, 0, 0) };
        if (!string.IsNullOrWhiteSpace(rendered.TrackText))
            chipWrap.Children.Add(BuildMetadataChip("track", rendered.TrackText, accent));
        if (!string.IsNullOrWhiteSpace(rendered.ResponsePhaseText))
            chipWrap.Children.Add(BuildMetadataChip("phase", rendered.ResponsePhaseText, accent));
        if (!string.IsNullOrWhiteSpace(rendered.IntentsText))
            chipWrap.Children.Add(BuildMetadataChip("concepts", rendered.IntentsText, accent));
        if (!string.IsNullOrWhiteSpace(rendered.ActionGramsText))
            chipWrap.Children.Add(BuildMetadataChip("actions", rendered.ActionGramsText, accent));
        if (!string.IsNullOrWhiteSpace(rendered.LearningText))
            chipWrap.Children.Add(BuildMetadataChip("learning", rendered.LearningText, accent));
        if (chipWrap.Children.Count > 0)
            bubbleStack.Children.Add(chipWrap);

        if (rendered.PlanVerbs.Count > 0)
        {
            var verbWrap = new WrapPanel { Margin = new Thickness(0, 6, 0, 0) };
            foreach (var verb in rendered.PlanVerbs)
            {
                var capturedVerb = verb;
                var button = new Button
                {
                    Content = string.Equals(capturedVerb, "execute", StringComparison.OrdinalIgnoreCase)
                        ? "Allow execution" : capturedVerb,
                    Tag = capturedVerb,
                    Margin = new Thickness(0, 0, 6, 4),
                    Padding = new Thickness(8, 3, 8, 3),
                    Foreground = Brushes.White,
                    Background = new SolidColorBrush(Color.FromRgb(24, 62, 88)),
                    BorderBrush = accent
                };
                button.Click += async (_, __) =>
                {
                    var goal = string.IsNullOrWhiteSpace(rendered.PlanGoal) ? rendered.BotText : rendered.PlanGoal;
                    AppendTranscript($"> [plan:{capturedVerb}] {goal}");
                    await ExecuteActionAsync("aiml", $"{capturedVerb} {goal}");
                };
                verbWrap.Children.Add(button);
            }
            bubbleStack.Children.Add(verbWrap);
        }

        bubbleStack.Children.Add(new TextBlock
        {
            Text = DateTime.Now.ToString("HH:mm:ss"),
            Foreground = new SolidColorBrush(Color.FromRgb(145, 164, 184)),
            FontSize = 11,
            Margin = new Thickness(0, 4, 0, 0)
        });

        bubble.Child = bubbleStack;
        layout.Children.Add(bubble);

        var card = new Border
        {
            Margin = new Thickness(0, 0, 0, 8),
            Child = layout
        };
        if (animatedSprite is not null)
        {
            card.Tag = animatedSprite;
            _animatedSprites.Add(animatedSprite);
            EnsureAvatarAnimationTimer();
        }

        BotCardsPanel.Children.Add(card);
        while (BotCardsPanel.Children.Count > MaxBotCards)
        {
            if (BotCardsPanel.Children[0] is Border oldCard && oldCard.Tag is AnimatedSpriteState oldAnimatedSprite)
                _animatedSprites.Remove(oldAnimatedSprite);
            BotCardsPanel.Children.RemoveAt(0);
        }
        BotCardsScrollViewer?.ScrollToEnd();
    }

    private sealed class RenderedResponse
    {
        public sealed class TimerDirective
        {
            public bool Clear { get; init; }
            public string TimerId { get; init; } = string.Empty;
            public int RemainingSeconds { get; init; } = -1;
            public bool HasTimer => !Clear && !string.IsNullOrWhiteSpace(TimerId) && RemainingSeconds > 0;
        }

        public string BotText { get; init; } = string.Empty;
        public string Source { get; init; } = string.Empty;
        public string Status { get; init; } = string.Empty;
        public string FrameKey { get; init; } = "default";
        public string ResponsePhaseText { get; init; } = string.Empty;
        public string TrackText { get; init; } = string.Empty;
        public string IntentsText { get; init; } = string.Empty;
        public string ActionGramsText { get; init; } = string.Empty;
        public string LearningText { get; init; } = string.Empty;
        public string NgramsText { get; init; } = string.Empty;
        public string PlanCardText { get; init; } = string.Empty;
        public string PlanGoal { get; init; } = string.Empty;
        public List<string> PlanVerbs { get; init; } = new();
        public string DisplayText { get; init; } = string.Empty;
        public TimerDirective Timer { get; init; } = new();
    }

    private static string ReadString(JsonElement obj, string propertyName)
    {
        if (obj.ValueKind != JsonValueKind.Object) return string.Empty;
        if (!obj.TryGetProperty(propertyName, out var prop)) return string.Empty;
        return prop.ValueKind == JsonValueKind.String ? (prop.GetString() ?? string.Empty) : string.Empty;
    }

    private static string ReadNestedString(JsonElement obj, string parent, string child)
    {
        if (obj.ValueKind != JsonValueKind.Object) return string.Empty;
        if (!obj.TryGetProperty(parent, out var parentProp) || parentProp.ValueKind != JsonValueKind.Object) return string.Empty;
        return ReadString(parentProp, child);
    }

    private static int ReadInt(JsonElement obj, string propertyName)
    {
        if (obj.ValueKind != JsonValueKind.Object) return -1;
        if (!obj.TryGetProperty(propertyName, out var prop)) return -1;
        if (prop.ValueKind == JsonValueKind.Number && prop.TryGetInt32(out var i))
            return i;
        if (prop.ValueKind == JsonValueKind.Number)
            return (int)Math.Round(prop.GetDouble());
        if (prop.ValueKind == JsonValueKind.String && int.TryParse(prop.GetString(), out var parsed))
            return parsed;
        return -1;
    }

    private static void CollectStrings(JsonElement maybeArray, ISet<string> sink)
    {
        if (maybeArray.ValueKind != JsonValueKind.Array) return;
        foreach (var item in maybeArray.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String) continue;
            var value = item.GetString() ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(value))
                sink.Add(value.Trim());
        }
    }

    private static string ReadPlanCard(JsonElement result)
    {
        if (result.ValueKind != JsonValueKind.Object || !result.TryGetProperty("plan_card", out var card) || card.ValueKind != JsonValueKind.Object)
            return string.Empty;
        var lines = new List<string>();
        foreach (var key in new[] { "goal", "owner", "phase", "status", "next", "authority" })
        {
            var value = ReadString(card, key);
            if (!string.IsNullOrWhiteSpace(value)) lines.Add($"{key}: {value}");
        }
        if (card.TryGetProperty("tasks", out var tasks) && tasks.ValueKind == JsonValueKind.Array)
        {
            var index = 1;
            foreach (var task in tasks.EnumerateArray())
            {
                if (task.ValueKind != JsonValueKind.Object) continue;
                var text = ReadString(task, "text");
                var status = ReadString(task, "status");
                if (!string.IsNullOrWhiteSpace(text))
                    lines.Add($"task {index++}: {text}" + (string.IsNullOrWhiteSpace(status) ? string.Empty : $" [{status}]"));
            }
        }
        return string.Join(Environment.NewLine, lines);
    }

    private static (string Goal, List<string> Verbs) ReadPlanActions(JsonElement result)
    {
        var verbs = new List<string>();
        if (result.ValueKind != JsonValueKind.Object || !result.TryGetProperty("plan_card", out var card) || card.ValueKind != JsonValueKind.Object)
            return (string.Empty, verbs);
        var goal = ReadString(card, "goal");
        if (card.TryGetProperty("actions", out var actions) && actions.ValueKind == JsonValueKind.Array)
        {
            foreach (var action in actions.EnumerateArray())
            {
                if (action.ValueKind != JsonValueKind.String) continue;
                var value = action.GetString() ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(value)) verbs.Add(value.Trim());
            }
        }
        return (goal, verbs);
    }

    private static void CollectNgramReferences(JsonElement result, ISet<string> sink)
    {
        if (result.ValueKind != JsonValueKind.Object) return;
        if (!result.TryGetProperty("ngram_references", out var refs) || refs.ValueKind != JsonValueKind.Array) return;
        foreach (var row in refs.EnumerateArray())
        {
            if (row.ValueKind != JsonValueKind.Object) continue;
            var gram = ReadString(row, "gram");
            if (!string.IsNullOrWhiteSpace(gram))
                sink.Add(gram.Trim());
        }
    }

    private static void CollectNodeMeaningKeys(JsonElement result, ISet<string> sink)
    {
        if (result.ValueKind != JsonValueKind.Object) return;
        if (!result.TryGetProperty("nodes", out var nodes) || nodes.ValueKind != JsonValueKind.Array) return;
        foreach (var node in nodes.EnumerateArray())
        {
            if (node.ValueKind != JsonValueKind.Object) continue;
            var meaningKey = ReadString(node, "meaning_key");
            if (!string.IsNullOrWhiteSpace(meaningKey))
                sink.Add(meaningKey.Trim());
        }
    }

    private static void CollectActionGramRows(JsonElement result, ISet<string> sink)
    {
        if (result.ValueKind != JsonValueKind.Object) return;
        if (!result.TryGetProperty("action_grams", out var rows) || rows.ValueKind != JsonValueKind.Array) return;
        foreach (var row in rows.EnumerateArray())
        {
            if (row.ValueKind != JsonValueKind.Object) continue;
            var gram = ReadString(row, "gram");
            var op = ReadString(row, "op");
            var intent = ReadString(row, "intent");
            if (string.IsNullOrWhiteSpace(gram) && string.IsNullOrWhiteSpace(intent) && string.IsNullOrWhiteSpace(op))
                continue;

            var label = gram;
            if (!string.IsNullOrWhiteSpace(op))
                label = string.IsNullOrWhiteSpace(label) ? op : $"{label} => {op}";
            else if (!string.IsNullOrWhiteSpace(intent))
                label = string.IsNullOrWhiteSpace(label) ? intent : $"{label} => {intent}";

            if (!string.IsNullOrWhiteSpace(label))
                sink.Add(label.Trim());
        }
    }

    private static void CollectLearnedFacts(JsonElement result, ISet<string> sink)
    {
        if (result.ValueKind != JsonValueKind.Object) return;
        if (!result.TryGetProperty("learned_facts", out var rows) || rows.ValueKind != JsonValueKind.Array) return;
        foreach (var row in rows.EnumerateArray())
        {
            if (row.ValueKind != JsonValueKind.Object) continue;
            var subject = ReadString(row, "subject");
            var predicate = ReadString(row, "predicate");
            if (string.IsNullOrWhiteSpace(subject) || string.IsNullOrWhiteSpace(predicate))
                continue;
            sink.Add($"{subject} => {predicate}");
        }
    }

    private static RenderedResponse.TimerDirective ParseTimerDirective(JsonElement result)
    {
        if (result.ValueKind != JsonValueKind.Object) return new RenderedResponse.TimerDirective();
        var source = ReadString(result, "source");
        if (!string.Equals(source, "utility-runtime", StringComparison.OrdinalIgnoreCase))
            return new RenderedResponse.TimerDirective();

        var status = ReadString(result, "status");
        if (string.Equals(status, "timer_started", StringComparison.OrdinalIgnoreCase))
        {
            var timerId = ReadString(result, "timer_id");
            var duration = ReadInt(result, "duration_seconds");
            if (!string.IsNullOrWhiteSpace(timerId) && duration > 0)
            {
                return new RenderedResponse.TimerDirective
                {
                    TimerId = timerId.Trim(),
                    RemainingSeconds = duration
                };
            }
            return new RenderedResponse.TimerDirective();
        }

        if (string.Equals(status, "timer_status", StringComparison.OrdinalIgnoreCase))
        {
            var activeCount = ReadInt(result, "active_count");
            if (activeCount == 0)
                return new RenderedResponse.TimerDirective { Clear = true };

            if (result.TryGetProperty("timers", out var rows) && rows.ValueKind == JsonValueKind.Array)
            {
                string timerId = string.Empty;
                var remaining = -1;
                foreach (var row in rows.EnumerateArray())
                {
                    if (row.ValueKind != JsonValueKind.Object) continue;
                    var state = ReadString(row, "state");
                    if (!string.Equals(state, "running", StringComparison.OrdinalIgnoreCase)) continue;
                    var rowRemaining = ReadInt(row, "remaining_seconds");
                    var rowTimerId = ReadString(row, "timer_id");
                    if (rowRemaining < 0 || string.IsNullOrWhiteSpace(rowTimerId)) continue;
                    if (rowRemaining > remaining)
                    {
                        remaining = rowRemaining;
                        timerId = rowTimerId.Trim();
                    }
                }

                if (!string.IsNullOrWhiteSpace(timerId) && remaining > 0)
                {
                    return new RenderedResponse.TimerDirective
                    {
                        TimerId = timerId,
                        RemainingSeconds = remaining
                    };
                }
            }
        }

        return new RenderedResponse.TimerDirective();
    }

    private static RenderedResponse RenderMx2lmResponse(string outText)
    {
        if (string.IsNullOrWhiteSpace(outText))
            return new RenderedResponse();

        try
        {
            using var doc = JsonDocument.Parse(outText);
            var root = doc.RootElement;
            var result = root;
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("result", out var resultProp))
                result = resultProp;

            var botText = ReadString(result, "text");
            if (string.IsNullOrWhiteSpace(botText))
                botText = ReadString(result, "response");
            if (string.Equals(botText, "<mx2lm>{ngrams}</mx2lm>", StringComparison.OrdinalIgnoreCase) ||
                string.IsNullOrWhiteSpace(botText))
            {
                botText = ReadNestedString(result, "structured_composition", "sentence");
            }
            if (string.IsNullOrWhiteSpace(botText))
                botText = ReadNestedString(result, "structured_composition", "paragraph");
            if (string.IsNullOrWhiteSpace(botText))
                botText = "Response received.";

            var timerDirective = ParseTimerDirective(result);

            var source = ReadString(result, "source");
            var status = ReadString(result, "status");
            var responsePhase = ReadString(result, "response_phase");
            var invoke = ReadString(result, "invoke");
            var trackText = string.Empty;
            if (!string.IsNullOrWhiteSpace(source) || !string.IsNullOrWhiteSpace(invoke))
            {
                if (string.IsNullOrWhiteSpace(invoke)) trackText = source;
                else if (string.IsNullOrWhiteSpace(source)) trackText = invoke;
                else trackText = $"{source} · {invoke}";
            }

            var grams = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var intents = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var actionGrams = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var learnedFacts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (result.ValueKind == JsonValueKind.Object)
            {
                if (result.TryGetProperty("bigrams", out var bi))
                    CollectStrings(bi, grams);
                if (result.TryGetProperty("trigrams", out var tri))
                    CollectStrings(tri, grams);
                if (result.TryGetProperty("intent_fragments", out var intentParts))
                    CollectStrings(intentParts, intents);
                if (!string.IsNullOrWhiteSpace(responsePhase))
                    intents.Add($"phase:{responsePhase.Trim().ToLowerInvariant()}");
                CollectActionGramRows(result, actionGrams);
                CollectLearnedFacts(result, learnedFacts);
                CollectNgramReferences(result, grams);
                if (grams.Count == 0)
                    CollectNodeMeaningKeys(result, grams);
            }
            var intentsText = intents.Count > 0 ? string.Join(" | ", intents) : string.Empty;
            if (!string.IsNullOrWhiteSpace(intentsText))
                intentsText = Ngrams.JoinConcepts(intents);
            var actionGramsText = Ngrams.JoinActions(actionGrams);
            var learningText = Ngrams.JoinLearning(learnedFacts);
            var ngramsText = Ngrams.JoinNgrams(grams);
            var planCardText = ReadPlanCard(result);
            var planActions = ReadPlanActions(result);
            var frameKey = Ngrams.ResolveSvgCFrame(source, status, intents, actionGrams);

            var display = botText;
            if (!string.IsNullOrWhiteSpace(trackText))
                display += Environment.NewLine + $"track: {trackText}";
            if (!string.IsNullOrWhiteSpace(responsePhase))
                display += Environment.NewLine + $"phase: {responsePhase}";
            if (!string.IsNullOrWhiteSpace(intentsText))
                display += Environment.NewLine + $"concepts: {intentsText}";
            if (!string.IsNullOrWhiteSpace(actionGramsText))
                display += Environment.NewLine + $"actions: {actionGramsText}";
            if (!string.IsNullOrWhiteSpace(learningText))
                display += Environment.NewLine + $"learning: {learningText}";
            if (!string.IsNullOrWhiteSpace(planCardText))
                display += Environment.NewLine + "tasks:" + Environment.NewLine + planCardText;
            display += Environment.NewLine + $"ngrams: {ngramsText}";

            return new RenderedResponse
            {
                BotText = botText,
                Source = source,
                Status = status,
                FrameKey = frameKey,
                ResponsePhaseText = responsePhase,
                TrackText = trackText,
                IntentsText = intentsText,
                ActionGramsText = actionGramsText,
                LearningText = learningText,
                NgramsText = ngramsText,
                PlanCardText = planCardText,
                PlanGoal = planActions.Goal,
                PlanVerbs = planActions.Verbs,
                DisplayText = display,
                Timer = timerDirective
            };
        }
        catch
        {
            var line = Json.FirstLine(outText);
            return new RenderedResponse
            {
                BotText = line,
                DisplayText = Json.Pretty(outText)
            };
        }
    }

    private async Task ExecuteActionAsync(string action, string query)
    {
        if (_isRunning)
        {
            return;
        }

        if (!File.Exists(_commandShim))
        {
            MessageBox.Show($"Command shim not found: {_commandShim}", "Missing command shim",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        _isRunning = true;
        SendButton.IsEnabled = false;
        ClearButton.IsEnabled = false;
        try
        {
            var result = await RunCommandAsync(action, query);
            var outText = (result.StdOut ?? string.Empty).Trim();
            var errText = (result.StdErr ?? string.Empty).Trim();

            if (!string.IsNullOrWhiteSpace(outText))
            {
                var rendered = RenderMx2lmResponse(outText);
                SetRenderedOutput(rendered.DisplayText);
                AppendTranscript("[bot] " + rendered.BotText);
                if (!string.IsNullOrWhiteSpace(rendered.TrackText))
                    AppendTranscript("[track] " + rendered.TrackText);
                if (!string.IsNullOrWhiteSpace(rendered.ResponsePhaseText))
                    AppendTranscript("[phase] " + rendered.ResponsePhaseText);
                if (!string.IsNullOrWhiteSpace(rendered.IntentsText))
                    AppendTranscript("[concepts] " + rendered.IntentsText);
                if (!string.IsNullOrWhiteSpace(rendered.ActionGramsText))
                    AppendTranscript("[actions] " + rendered.ActionGramsText);
                if (!string.IsNullOrWhiteSpace(rendered.LearningText))
                    AppendTranscript("[learning] " + rendered.LearningText);
                if (!string.IsNullOrWhiteSpace(rendered.NgramsText))
                    AppendTranscript("[ngrams] " + rendered.NgramsText);

                if (rendered.Timer.Clear)
                    StopCountdown();
                else if (rendered.Timer.HasTimer)
                    StartOrRefreshCountdown(rendered.Timer.TimerId, rendered.Timer.RemainingSeconds);

                AddBotCard(rendered);
            }
            else
            {
                SetRenderedOutput(string.Empty);
            }

            if (!string.IsNullOrWhiteSpace(errText))
            {
                AppendTranscript("[stderr] " + Json.FirstLine(errText));
            }
            else if (string.IsNullOrWhiteSpace(outText))
            {
                const string emptyMessage = "No response payload returned from runtime.";
                RawTextBox.Text = "{\"ok\":false,\"status\":\"empty_response\",\"message\":\"No response payload returned from runtime.\"}";
                AppendTranscript("[stderr] " + emptyMessage);
            }

            AppendTranscript($"[exit] {result.ExitCode}");
        }
        catch (Exception ex)
        {
            AppendTranscript("[error] " + ex.Message);
        }
        finally
        {
            _isRunning = false;
            SendButton.IsEnabled = true;
            ClearButton.IsEnabled = true;
        }
    }

    private async void SendButton_OnClick(object sender, RoutedEventArgs e)
    {
        var query = InputTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(query))
        {
            return;
        }

        var action = CurrentAction();
        AppendTranscript($"> [{action}] {query}");
        InputTextBox.Clear();
        InputTextBox.Focus();
        await ExecuteActionAsync(action, query);
    }

    private void ClearButton_OnClick(object sender, RoutedEventArgs e)
    {
        StopCountdown();
        _avatarAnimationTimer?.Stop();
        _animatedSprites.Clear();
        InputTextBox.Clear();
        TranscriptTextBox.Clear();
        BotCardsPanel.Children.Clear();
        RawTextBox.Clear();
        _lastRenderedOutput = string.Empty;
    }

    private async void InputTextBox_OnPreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && Keyboard.Modifiers.HasFlag(ModifierKeys.Control))
        {
            e.Handled = true;
            await Task.Yield();
            SendButton_OnClick(sender, e);
        }
    }

    private void ModeComboBox_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        RenderCommandPreview();
    }
}
'@

if ($includeDatasetRowsBool) {
    Write-GeneratedFile -RelativePath "dataset\toolcalls.jsonl" -Content @"
{"tool":"Agent","arg":"python bots.py center \"Build an app shell\"","text":"run center action through local bundle bot shim"}
{"tool":"Skill","arg":"powershell -File scripts\\launch-wpf-cli.ps1","text":"launch WPF terminal-plus-console shell"}
{"tool":"Command","arg":"powershell -File agents\\commands.ps1 -Action aiml -QueryBase64 c3RhdHVz -RawJson","text":"run AIML-first status query with runtime auto-start"}
{"tool":"Tool","arg":"rpc.key_query","text":"query runtime STATUS: lane via KHANARY RPC"}
"@
}

$duplicateInventory = New-BundleDuplicateInventory -RootPath $bundleRoot -ExcludeRelativePaths @(
    "bundle_duplicate_files.json",
    "mx2_builder_manifest.json"
)
Write-GeneratedFile -RelativePath "bundle_duplicate_files.json" -Content ($duplicateInventory | ConvertTo-Json -Depth 10)

$manifest = [ordered]@{
    schema = "mx2.bundle.manifest.v1"
    bundle = $BundleName
    generated_utc = $utcNow
    builder = (Resolve-Path $PSCommandPath).Path
    target_root = $bundleRoot
    source_policy = [ordered]@{
        repo_local_only = $true
        public_html_used = $false
    }
    duplicate_inventory = "bundle_duplicate_files.json"
    duplicate_summary = [ordered]@{
        file_count = [int]$duplicateInventory.file_count
        duplicate_group_count = [int]$duplicateInventory.duplicate_group_count
        duplicate_file_count = [int]$duplicateInventory.duplicate_file_count
    }
    copied_sources = [object[]]$copied
    generated_files = [string[]]$generated
}
Write-GeneratedFile -RelativePath "mx2_builder_manifest.json" -Content ($manifest | ConvertTo-Json -Depth 10)

Write-Host "Built MX2-style bundle at: $bundleRoot"
Write-Host "Generated files: $($generated.Count)"
Write-Host "Mirrored source files: $($copied.Count)"
