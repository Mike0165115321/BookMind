#!/bin/bash
# fix_ollama_network.sh - Automated Ollama configuration for WSL users

echo "------------------------------------------------"
echo "🛠️  Ollama WSL Network Fixer"
echo "------------------------------------------------"

# 1. Set OLLAMA_HOST=0.0.0.0 in Windows Environment Variables
echo "🚀 Setting OLLAMA_HOST=0.0.0.0 in Windows..."
powershell.exe -Command "[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0', 'User')"

# 2. Add Firewall Rule (Triggers UAC prompt on Windows)
echo "🛡️  Requesting Firewall Rule in Windows (Please check your Windows taskbar for UAC prompt)..."
powershell.exe -Command "Start-Process powershell -ArgumentList 'New-NetFirewallRule -DisplayName \"Allow Ollama for WSL\" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434 -Force' -Verb RunAs"

echo ""
echo "✅ Configuration sent to Windows!"
echo "------------------------------------------------"
echo "⚠️  CRITICAL STEP:"
echo "1. Go to your Windows System Tray (bottom right corner)."
echo "2. Right-click the Ollama icon and select 'Quit Ollama'."
echo "3. Start Ollama again from your Start Menu."
echo "------------------------------------------------"
echo "After that, run './run_all.sh' in WSL and it will work!"
