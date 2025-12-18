import os
import json
import requests
import urllib3
import argparse

# --- Config -----------------------------------------------------------------

# Where to read the token from (env var override, else default path)
token_path = os.environ.get(
    "TOKEN_FILE",
    os.path.expanduser("~/Repositories/xai-playground/token.txt"),
)

# Where to save traces (can override with TRACE_OUTPUT_DIR)
output_dir = os.environ.get("TRACE_OUTPUT_DIR", "traces")
os.makedirs(output_dir, exist_ok=True)

url = "https://expressif.cea.fr:5100/api/sync/process?rulebase=tip.rules&querier=trace"

parser = argparse.ArgumentParser(
    description="Send trace request with food/service scores (0-10)."
)
parser.add_argument(
    "-f", "--food", type=int, choices=range(0, 11), default=8,
    help="Food score (0-10). Default: 8"
)
parser.add_argument(
    "-s", "--service", type=int, choices=range(0, 11), default=7,
    help="Service score (0-10). Default: 7"
)
args = parser.parse_args()

food_score = args.food
service_score = args.service

# --- Token loading -----------------------------------------------------------

try:
    with open(token_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
except Exception as e:
    raise SystemExit(f"Unable to read token from {token_path}: {e}")

if not token:
    raise SystemExit(f"Token file {token_path} is empty.")

# --- Request -----------------------------------------------------------------

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json",
}

payload = {
    "inputValues": [
        {"name": "food", "value": food_score},
        {"name": "service", "value": service_score},
    ]
}

# (Temporary) ignore SSL verification because the cert is expired on the server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

response = requests.post(url, headers=headers, json=payload, verify=False)

print("Status:", response.status_code)

try:
    data = response.json()
except json.JSONDecodeError:
    print("Response is not valid JSON:\n", response.text)
    raise SystemExit(1)

# --- Save JSON ---------------------------------------------------------------

output_name = f"tip_trace_food{food_score}_service{service_score}.json"
output_path = os.path.join(output_dir, output_name)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved JSON trace to: {output_path}")
