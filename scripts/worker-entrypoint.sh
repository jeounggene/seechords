#!/bin/sh
# Start the PO token server in the background, wait for it, then run the worker.

bgutil-pot server --port 4416 &
POT_PID=$!

# Wait for the server to be ready (up to 10s)
for i in $(seq 1 20); do
  if curl -sf http://127.0.0.1:4416/ping >/dev/null 2>&1; then
    echo "[Entrypoint] PO token server ready"
    break
  fi
  sleep 0.5
done

if ! curl -sf http://127.0.0.1:4416/ping >/dev/null 2>&1; then
  echo "[Entrypoint] WARNING: PO token server not responding after 10s, proceeding anyway"
fi

python worker.py
EXIT_CODE=$?

kill $POT_PID 2>/dev/null
exit $EXIT_CODE
