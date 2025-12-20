from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import time
import json

PORT = 8001
STATS_FILE = "/mnt/c/Programming/Python/qos/visualiser/pox_stats.json"

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return  # silence logs

    def _safe_read_json(self):
        """
        Safely read the POX stats file:
        - Retry until JSON is complete
        - Avoid partial reads
        """
        for _ in range(5):  # try up to 5 times
            try:
                with open(STATS_FILE, "r") as f:
                    data = f.read().strip()

                # If file empty or not fully written → retry
                if not data:
                    time.sleep(0.05)
                    continue

                # Try parsing JSON
                try:
                    json.loads(data)
                    return data  # valid JSON
                except json.JSONDecodeError:
                    # Probably being written → retry
                    time.sleep(0.05)
                    continue

            except FileNotFoundError:
                return "{}"  # file not created yet

        return "{}"  # give up after retries

    def do_GET(self):
        if self.path == "/stats":
            data = self._safe_read_json()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                self.wfile.write(data.encode())
            except BrokenPipeError:
                pass

        else:
            self.send_response(404)
            self.end_headers()


print(f"[INFO] Stats Proxy running → http://localhost:{PORT}/stats")
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
