#!/bin/bash
# run.sh
#
# Launches both the Bokeh server (SINDy module backend) and the Flask app
# (frontend) with a single command, instead of needing two split terminals.
#
# Usage:
#   chmod +x run.sh   (only needed once)
#   ./run.sh
#
# Press Ctrl+C once to stop BOTH processes cleanly.

echo "🚀 Starting Bokeh server..."
bokeh serve --allow-websocket-origin=127.0.0.1:8080 main.py &
BOKEH_PID=$!

echo "🚀 Starting Flask app..."
python3 flask_app.py &
FLASK_PID=$!

# When Ctrl+C is pressed, kill both background processes before exiting.
trap "echo '🛑 Stopping...'; kill $BOKEH_PID $FLASK_PID; exit" SIGINT SIGTERM

# Wait for both processes — script stays alive until either exits or Ctrl+C.
wait $BOKEH_PID $FLASK_PID