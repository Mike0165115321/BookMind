# 🛠️ Ollama WSL Networking Fix (Developer Documentation)

## 📌 Overview
This document explains the technical mechanism behind the `fix_ollama_network.sh` script and why it is necessary for developers using Ollama on Windows while running the BookMind suite in WSL (Windows Subsystem for Linux).

## 🚨 The Problem: Network Isolation
By default, **Ollama for Windows** only listens on `127.0.0.1:11434` (localhost).
In a WSL2 environment:
1. `localhost` inside WSL refers to the Linux VM, not the Windows Host.
2. Windows considers WSL as a separate network entity, thus blocking requests to `11434` via the Windows Firewall.
3. Ollama rejects any requests that do not originate from `127.0.0.1` unless configured otherwise.

## ⚡ The Solution: `fix_ollama_network.sh`
The script automates two critical configuration steps by calling Windows PowerShell directly from WSL.

### 1. Environment Variable: `OLLAMA_HOST`
- **Command**: `[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0', 'User')`
- **Action**: Sets a User-level environment variable in Windows.
- **Result**: Instructs Ollama to bind to all network interfaces (`0.0.0.0`), allowing it to accept connections from the WSL virtual network.

### 2. Windows Firewall Rule
- **Command**: `New-NetFirewallRule -DisplayName "Allow Ollama for WSL" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434`
- **Action**: Creates an Inbound rule in Windows Advanced Firewall.
- **Result**: Permits TCP traffic on port `11434`. Without this, Windows drops packets coming from the WSL IP range.

## 🛡️ Security Considerations
> [!WARNING]
> Setting `OLLAMA_HOST=0.0.0.0` makes your local Ollama API accessible to **anyone on your local network (WiFi/LAN)**.
> 
> - **In a trusted home/office network**: This is generally safe.
> - **In public WiFi**: This could allow others to use your GPU/Models. 
> - **To Revert**: Delete the `OLLAMA_HOST` variable from Windows Environment Variables and remove the Firewall rule.

## 🔍 Verification (For Developers)
To verify if the fix worked, run these commands from Windows:

1. **Check Binding**: `netstat -ano | findstr 11434`
   - Expected: `0.0.0.0:11434` (LISTENING)
2. **Test from WSL**: `nc -zv <Windows_Host_IP> 11434`
   - Expected: `Connection to <IP> 11434 port [tcp/*] succeeded!`

## 🔄 Why "Auto-Discovery" in Code?
The `OllamaClient` in this project uses an auto-discovery logic that attempts to find the Windows Host IP by parsing `ip route` (Gateway IP). This script ensures that once the IP is found, the connection is actually allowed by the host.

---
*Created by Antigravity AI for the BookMind Project.*
