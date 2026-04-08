#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Local Docker test that mirrors what the hackathon validator does.
#
# Usage:
#   chmod +x test_docker.sh
#   ./test_docker.sh
#
# What it does:
#   1. Starts a mock LiteLLM proxy on localhost:9999 that:
#      - Records every incoming request (proves calls reach API_BASE_URL)
#      - Returns a valid OpenAI-shaped JSON response
#   2. Builds the Docker image
#   3. Runs inference.py inside the container with injected env vars
#      (exactly as the validator would)
#   4. Shows whether [STEP] lines appear and whether the proxy was hit
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

PROXY_PORT=9999
IMAGE_NAME="content-moderation-env-test"
LOG_FILE="/tmp/mock_proxy_requests.log"

# ── 1. Start mock proxy ───────────────────────────────────────────
echo "▶ Starting mock LiteLLM proxy on port $PROXY_PORT ..."
rm -f "$LOG_FILE"

python3 - <<'PROXY_EOF' &
import http.server, json, datetime, os, sys

PORT = 9999
LOG  = "/tmp/mock_proxy_requests.log"

FAKE_RESPONSE = json.dumps({
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "test-model",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": '{"type":"moderate","post_id":"e1","decision":"remove","rationale":"P1"}'
        },
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
})

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode("utf-8", errors="replace")
        ts     = datetime.datetime.now().isoformat()
        line   = f"[{ts}] POST {self.path} | auth={self.headers.get('Authorization','NONE')[:40]}\n"
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(FAKE_RESPONSE.encode())

    def log_message(self, fmt, *args):
        pass  # silence default access log

with http.server.HTTPServer(("0.0.0.0", PORT), Handler) as srv:
    print(f"Mock proxy listening on :{PORT}", flush=True)
    srv.serve_forever()
PROXY_EOF

PROXY_PID=$!
sleep 1   # give it a moment to start

# ── 2. Build Docker image ─────────────────────────────────────────
echo "▶ Building Docker image ($IMAGE_NAME) ..."
docker build -t "$IMAGE_NAME" . 2>&1 | tail -5

# ── 3. Get host IP reachable from inside container ────────────────
# On Linux: use docker0 bridge (172.17.0.1)
# On Mac Docker Desktop: use host.docker.internal
HOST_IP=$(docker run --rm "$IMAGE_NAME" sh -c \
  "getent hosts host.docker.internal 2>/dev/null | awk '{print \$1}' || echo '172.17.0.1'")
echo "▶ Container will reach host at: $HOST_IP"

PROXY_URL="http://${HOST_IP}:${PROXY_PORT}/v1"
echo "▶ Injecting API_BASE_URL=$PROXY_URL"

# ── 4. Run inference.py inside the container ──────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo " Running inference.py inside Docker"
echo "═══════════════════════════════════════════"

# Use --add-host so host.docker.internal works on Linux too
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -e HF_TOKEN="test-proxy-key-abc123" \
  -e API_BASE_URL="$PROXY_URL" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e ENV_BASE_URL="https://saheb-content-moderation-env.hf.space" \
  "$IMAGE_NAME" \
  python inference.py 2>/tmp/inference_stderr.log
EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════"
echo " Results"
echo "═══════════════════════════════════════════"
echo "Exit code: $EXIT_CODE"
echo ""
echo "── Proxy hits (should be > 0) ─────────────"
if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    echo "✅ Mock proxy WAS called:"
    cat "$LOG_FILE"
else
    echo "❌ Mock proxy was NEVER called — calls did not reach API_BASE_URL"
fi

echo ""
echo "── stderr (first 40 lines) ─────────────────"
head -40 /tmp/inference_stderr.log

# ── 5. Cleanup ────────────────────────────────────────────────────
kill $PROXY_PID 2>/dev/null || true
echo ""
echo "Done."
