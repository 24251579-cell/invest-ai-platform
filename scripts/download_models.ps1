# Download sentiment checkpoints with curl (resume + retries) into C:\hf_models\<repo-name>.
# Used instead of huggingface_hub on unstable networks: curl -C - continues partial files.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1

$ErrorActionPreference = "Stop"
$Root = "C:\hf_models"
$Models = @{
    "blanchefort/rubert-base-cased-sentiment" = @("config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors")
    "ProsusAI/finbert" = @("config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors")
}

foreach ($repo in $Models.Keys) {
    $dir = Join-Path $Root ($repo.Split("/")[1])
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $Models[$repo]) {
        $out = Join-Path $dir $file
        $url = "https://huggingface.co/$repo/resolve/main/$file"
        $expected = $null
        $head = curl.exe -sIL $url | Select-String -Pattern '^(x-linked-size|content-length):\s*(\d+)' | Select -Last 1
        if ($head) { $expected = [int64]$head.Matches[0].Groups[2].Value }
        for ($i = 1; $i -le 500; $i++) {
            $have = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
            if ($expected -and $have -ge $expected) { break }
            Write-Host ("{0}/{1}: {2:N1} / {3:N1} MB, attempt {4}" -f $repo, $file, ($have/1MB), ($expected/1MB), $i)
            curl.exe -sL -C - --connect-timeout 20 --speed-limit 2000 --speed-time 30 -o $out $url
            if (-not $expected -and $LASTEXITCODE -eq 0) { break }
            Start-Sleep 3
        }
        $have = (Get-Item $out).Length
        if ($expected -and $have -ne $expected) { throw "$repo/$file incomplete: $have of $expected" }
        Write-Host ("OK {0}/{1} ({2:N1} MB)" -f $repo, $file, ($have/1MB))
    }
}
Write-Host "ALL_MODELS_DONE"
