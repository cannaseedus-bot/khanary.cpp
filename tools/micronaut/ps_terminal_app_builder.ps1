[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("chat", "integration", "ide", "wpf", "svg", "build-app")]
    [string]$Mode,

    [string]$BaseRoot = "",
    [string]$TargetRoot = "",
    [switch]$OpenEditor,
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThruArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq $PassThruArgs) {
    $PassThruArgs = @()
} elseif ($PassThruArgs -isnot [array]) {
    $PassThruArgs = @($PassThruArgs)
}

if ($PassThruArgs.Count -gt 0 -and $PassThruArgs[0] -eq "--%") {
    $tail = if ($PassThruArgs.Count -gt 1) { ($PassThruArgs[1..($PassThruArgs.Count - 1)] -join " ") } else { "" }
    if ([string]::IsNullOrWhiteSpace($tail)) {
        $PassThruArgs = @()
    } else {
        $PassThruArgs = ($tail -split '\s+') | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$defaultExampleBaseRoot = "C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18"
$defaultUiShellRoot = Join-Path $repoRoot ".ui-shell" # example/template workspace only
$legacyUiShellRoot = "C:\Users\canna\.ui-shell"
$defaultLocalBundleRoot = Join-Path $repoRoot "artifacts\micronaut-bundles\mx2lm-wpf-cli-v1"
$shellExe = if (Get-Command pwsh.exe -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }

function Resolve-BaseRootPath {
    if (-not [string]::IsNullOrWhiteSpace($BaseRoot)) {
        return $BaseRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($env:KUHUL_MX2_BASE)) {
        return $env:KUHUL_MX2_BASE
    }
    return $defaultExampleBaseRoot
}

function Is-TemplateWorkspacePath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    $normalized = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $templateRoots = @(
        [System.IO.Path]::GetFullPath($defaultUiShellRoot).TrimEnd('\'),
        [System.IO.Path]::GetFullPath($legacyUiShellRoot).TrimEnd('\')
    )
    foreach ($root in $templateRoots) {
        if ($normalized.Equals($root, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Get-LocalShimPath {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($TargetRoot)) {
        $targetAbs = [System.IO.Path]::GetFullPath($TargetRoot)
        if (-not (Is-TemplateWorkspacePath -Path $targetAbs)) {
            $candidates += (Join-Path $targetAbs "agents\commands.ps1")
        }
    }
    $candidates += (Join-Path $defaultLocalBundleRoot "agents\commands.ps1")

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Get-WpfLauncherPath {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($TargetRoot)) {
        $root = [System.IO.Path]::GetFullPath($TargetRoot)
        if (-not (Is-TemplateWorkspacePath -Path $root)) {
            $candidates += (Join-Path $root "scripts\launch-wpf-cli.ps1")
        }
    }
    $candidates += (Join-Path $defaultLocalBundleRoot "scripts\launch-wpf-cli.ps1")

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Require-Path {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not (Test-Path $Path)) {
        throw "$Message`nMissing path: $Path"
    }
}

function Get-EntrypointPath {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedBaseRoot,
        [Parameter(Mandatory = $true)][string]$SelectedMode
    )
    switch ($SelectedMode) {
        "chat" { return (Join-Path $ResolvedBaseRoot "micronaut-ui-chat-app.ps1") }
        "integration" { return (Join-Path $ResolvedBaseRoot "mx2_integration.ps1") }
        "svg" { return (Join-Path $ResolvedBaseRoot "kuhul.svg") }
        default { return "" }
    }
}

function Quote-Arg {
    param([string]$Text)
    if ($null -eq $Text) { return "" }
    if ($Text -match '\s|"') {
        return '"' + ($Text -replace '"', '\"') + '"'
    }
    return $Text
}

function Emit-CommandLine {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    $joined = ($Args | ForEach-Object { Quote-Arg $_ }) -join " "
    Write-Host ("$Program $joined")
}

$resolvedBaseRoot = Resolve-BaseRootPath

if ($Mode -eq "build-app") {
    if ([string]::IsNullOrWhiteSpace($TargetRoot)) {
        $TargetRoot = $defaultUiShellRoot
    }

    $lockedChat = Join-Path $resolvedBaseRoot "micronaut-ui-chat-app.ps1"
    $lockedIntegration = Join-Path $resolvedBaseRoot "mx2_integration.ps1"
    $lockedSvg = Join-Path $resolvedBaseRoot "kuhul.svg"
    $lockedBaseAvailable = (Test-Path $resolvedBaseRoot) -and
        (Test-Path $lockedChat) -and
        (Test-Path $lockedIntegration)

    $localShimPath = Get-LocalShimPath
    if (-not $lockedBaseAvailable -and [string]::IsNullOrWhiteSpace($localShimPath)) {
        throw "No launch surface available. Locked base entrypoints are missing and no local commands.ps1 shim was found."
    }

    $targetAbs = [System.IO.Path]::GetFullPath($TargetRoot)
    if ($lockedBaseAvailable -and $targetAbs.StartsWith([System.IO.Path]::GetFullPath($resolvedBaseRoot), [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "TargetRoot must not be inside the locked base bundle tree."
    }

    if ($DryRun) {
        Write-Host "Would create example/template app-builder folder at: $targetAbs"
        if ($lockedBaseAvailable) {
            Write-Host "Using read-only example base: $resolvedBaseRoot"
        } else {
            Write-Host "Using local shim fallback: $localShimPath"
        }
        return
    }

    $null = New-Item -ItemType Directory -Path $targetAbs -Force
    $assetsDir = Join-Path $targetAbs "assets"
    $scriptsDir = Join-Path $targetAbs "scripts"
    $null = New-Item -ItemType Directory -Path $assetsDir -Force
    $null = New-Item -ItemType Directory -Path $scriptsDir -Force

    if ($lockedBaseAvailable -and (Test-Path $lockedSvg)) {
        Copy-Item -LiteralPath $lockedSvg -Destination (Join-Path $assetsDir "kuhul.svg") -Force
    }

    if ($lockedBaseAvailable) {
        $launchChat = @'
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$baseRoot = "{BASE_ROOT}"
$entry = Join-Path $baseRoot "micronaut-ui-chat-app.ps1"
$env:KUHUL_SVG_PATH = Join-Path $PSScriptRoot "..\assets\kuhul.svg"
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }
& $shell -NoProfile -ExecutionPolicy Bypass -File $entry @Args
'@

        $launchIntegration = @'
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$baseRoot = "{BASE_ROOT}"
$entry = Join-Path $baseRoot "mx2_integration.ps1"
$env:KUHUL_SVG_PATH = Join-Path $PSScriptRoot "..\assets\kuhul.svg"
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }
& $shell -NoProfile -ExecutionPolicy Bypass -File $entry @Args
'@

        Set-Content -LiteralPath (Join-Path $scriptsDir "launch-chat.ps1") -Value ($launchChat.Replace("{BASE_ROOT}", $resolvedBaseRoot)) -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $scriptsDir "launch-integration.ps1") -Value ($launchIntegration.Replace("{BASE_ROOT}", $resolvedBaseRoot)) -Encoding UTF8
    } else {
        $launchChat = @'
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$shim = "{LOCAL_SHIM}"
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }
& $shell -NoProfile -ExecutionPolicy Bypass -File $shim -Action aiml @Args
'@

        $launchIntegration = @'
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$shim = "{LOCAL_SHIM}"
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }
& $shell -NoProfile -ExecutionPolicy Bypass -File $shim -Action center @Args
'@
        Set-Content -LiteralPath (Join-Path $scriptsDir "launch-chat.ps1") -Value ($launchChat.Replace("{LOCAL_SHIM}", $localShimPath)) -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $scriptsDir "launch-integration.ps1") -Value ($launchIntegration.Replace("{LOCAL_SHIM}", $localShimPath)) -Encoding UTF8
    }

    $launchIde = @'
param([string]$Workspace = "")
if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }
Start-Process -FilePath $shell -ArgumentList @("-NoExit","-NoProfile") -WorkingDirectory $Workspace
if (Get-Command code -ErrorAction SilentlyContinue) {
    Start-Process -FilePath "code" -ArgumentList @($Workspace)
}
'@

    $launchWpf = @'
param([string]$Configuration = "Debug")
$candidate = "{WPF_LAUNCHER}"
if (-not (Test-Path $candidate)) {
    throw "WPF launcher not found: $candidate"
}
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh.exe" } else { "powershell.exe" }
& $shell -NoProfile -ExecutionPolicy Bypass -File $candidate -Configuration $Configuration
exit $LASTEXITCODE
'@

    Set-Content -LiteralPath (Join-Path $scriptsDir "launch-ide.ps1") -Value $launchIde -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $scriptsDir "launch-wpf-cli.ps1") -Value ($launchWpf.Replace("{WPF_LAUNCHER}", (Join-Path $defaultLocalBundleRoot "scripts\launch-wpf-cli.ps1"))) -Encoding UTF8

    $assets = @()
    if ($lockedBaseAvailable -and (Test-Path $lockedSvg)) {
        $assets += [ordered]@{
            source = $lockedSvg
            target = "assets/kuhul.svg"
        }
    }

    $manifest = [ordered]@{
        schema = "ps-terminal-app-builder.manifest.v1"
        generated_utc = (Get-Date).ToUniversalTime().ToString("o")
        base_root = if ($lockedBaseAvailable) { $resolvedBaseRoot } else { $null }
        example_root = $defaultExampleBaseRoot
        local_fallback_shim = if ([string]::IsNullOrWhiteSpace($localShimPath)) { $null } else { $localShimPath }
        target_root = $targetAbs
        workspace_role = "example-template"
        runtime_hosting = $false
        readonly_base = $true
        source_mode = if ($lockedBaseAvailable) { "locked-base" } else { "local-shim-fallback" }
        entrypoints = [ordered]@{
            chat = "scripts/launch-chat.ps1"
            integration = "scripts/launch-integration.ps1"
            ide = "scripts/launch-ide.ps1"
            wpf = "scripts/launch-wpf-cli.ps1"
        }
        assets = $assets
    }
    ($manifest | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $targetAbs "builder.manifest.json") -Encoding UTF8
    Write-Host "Built example/template app-builder workspace: $targetAbs"
    return
}

if ($Mode -eq "ide") {
    if ([string]::IsNullOrWhiteSpace($TargetRoot)) {
        $workspace = $repoRoot
    } else {
        $workspace = [System.IO.Path]::GetFullPath($TargetRoot)
    }
    if (-not (Test-Path $workspace)) {
        if ($DryRun) {
            Write-Host "Would create IDE workspace: $workspace"
        } else {
            $null = New-Item -ItemType Directory -Path $workspace -Force
        }
    }

    if ($DryRun) {
        Emit-CommandLine -Program $shellExe -Args @("-NoExit", "-NoProfile")
        if ($OpenEditor) {
            Emit-CommandLine -Program "code" -Args @($workspace)
        }
        return
    }

    Start-Process -FilePath $shellExe -ArgumentList @("-NoExit", "-NoProfile") -WorkingDirectory $workspace
    if ($OpenEditor -and (Get-Command code -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath "code" -ArgumentList @($workspace)
    }
    Write-Host "Opened IDE/terminal surface at: $workspace"
    return
}

if ($Mode -eq "wpf") {
    $wpfLauncher = Get-WpfLauncherPath
    if ([string]::IsNullOrWhiteSpace($wpfLauncher)) {
        throw "WPF launcher not found. Build or point to a bundle with scripts\launch-wpf-cli.ps1 inside this repo."
    }
    $invokeArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $wpfLauncher) + $PassThruArgs
    if ($DryRun) {
        Emit-CommandLine -Program $shellExe -Args $invokeArgs
        return
    }
    Write-Host "Launching WPF terminal app: $wpfLauncher"
    & $shellExe @invokeArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    return
}

$entrypoint = Get-EntrypointPath -ResolvedBaseRoot $resolvedBaseRoot -SelectedMode $Mode
$lockedEntrypointReady = (-not [string]::IsNullOrWhiteSpace($entrypoint)) -and (Test-Path $entrypoint)
$localShimPath = Get-LocalShimPath

if ($Mode -eq "svg") {
    if (-not $lockedEntrypointReady) {
        $svgCandidates = @(
            (Join-Path $defaultUiShellRoot "assets\kuhul.svg"),
            (Join-Path $legacyUiShellRoot "assets\kuhul.svg"),
            (Join-Path $defaultLocalBundleRoot "assets\kuhul.svg")
        )
        foreach ($cand in $svgCandidates) {
            if (Test-Path $cand) {
                $entrypoint = (Resolve-Path $cand).Path
                $lockedEntrypointReady = $true
                break
            }
        }
    }
    if (-not $lockedEntrypointReady) {
        throw "No KUHUL SVG asset found locally. Set -BaseRoot to an example tree containing kuhul.svg or place one under $defaultUiShellRoot\assets."
    }
    if ($DryRun) {
        Emit-CommandLine -Program "Start-Process" -Args @($entrypoint)
        return
    }
    Start-Process -FilePath $entrypoint
    Write-Host "Opened KUHUL identity asset: $entrypoint"
    return
}

$kuhulSvg = Join-Path $resolvedBaseRoot "kuhul.svg"
if (Test-Path $kuhulSvg) {
    $env:KUHUL_SVG_PATH = $kuhulSvg
}

if ($lockedEntrypointReady) {
    $invokeArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $entrypoint) + $PassThruArgs
    if ($DryRun) {
        Emit-CommandLine -Program $shellExe -Args $invokeArgs
        return
    }

    Write-Host "Launching $Mode entrypoint from read-only example base: $entrypoint"
    & $shellExe @invokeArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    return
}

if (($Mode -eq "chat" -or $Mode -eq "integration") -and (-not [string]::IsNullOrWhiteSpace($localShimPath))) {
    $shimAction = if ($Mode -eq "chat") { "aiml" } else { "center" }
    $invokeArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $localShimPath, "-Action", $shimAction) + $PassThruArgs
    if ($DryRun) {
        Emit-CommandLine -Program $shellExe -Args $invokeArgs
        return
    }
    Write-Host "Launching $Mode via local shim in this workspace: $localShimPath"
    & $shellExe @invokeArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    return
}

throw "No launchable surface found for mode '$Mode'. Checked example root '$resolvedBaseRoot' and bundle candidates under '$defaultLocalBundleRoot'."
