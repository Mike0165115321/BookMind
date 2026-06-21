#!/bin/bash

# BookMind Bundle Runner
# This script starts both the Web Server and the Discord Bot.

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment (venv) not found. Please run 'python3 -m venv venv' first."
    exit 1
fi

# 2. Cleanup function to stop all processes on Ctrl+C
cleanup() {
    echo ""
    echo "🛑 Stopping all services..."
    # Kill all background jobs started by this script
    pkill -P $$
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "------------------------------------------------"
echo "📚 Starting BookMind Integrated Suite..."
echo "------------------------------------------------"

# 3. Start Web Server in background
echo "🚀 Starting Web Server..."
python3 web_server.py &
WEB_PID=$!

# Wait for server to initialize
sleep 5

# 4. Start Discord Bot in foreground
echo "🤖 Starting Discord Bot..."
python3 discord_bot.py

# If discord_bot exits, cleanup everything
cleanup
