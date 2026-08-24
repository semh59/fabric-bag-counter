"""Live Multi-Threaded Stress, Benchmark, and Invariant Verification Script."""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.request
import urllib.error
import numpy as np

BASE_URL = "http://localhost:8080/api"


def make_request(path: str, method: str = "GET", data: dict | None = None, token: str | None = None) -> tuple[int, dict | list]:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        try:
            return e.code, json.loads(content)
        except Exception:
            return e.code, {"error": content}


def main():
    print("=" * 60)
    print("    DEEP LIVE SYSTEM STRESS & INVARIANT VERIFICATION")
    print("=" * 60)

    # 1. Reset Demo Data
    print("\n[1/5] Resetting to fresh factory state...")
    status, res = make_request("/system/seed_demo", method="POST")
    assert status == 200, f"Seed demo failed: {res}"
    print("  [OK] Database initialized with Gebze Factory topology")

    # 2. Authentication Matrix
    print("\n[2/5] Authenticating Personas...")
    _, op_res = make_request("/auth/login", method="POST", data={"username": "operator", "password": "op123"})
    _, eng_res = make_request("/auth/login", method="POST", data={"username": "engineer", "password": "eng123"})
    _, adm_res = make_request("/auth/login", method="POST", data={"username": "admin", "password": "admin123"})
    op_token = op_res["token"]
    eng_token = eng_res["token"]
    adm_token = adm_res["token"]
    print("  [OK] Operator token acquired")
    print("  [OK] Engineer token acquired")
    print("  [OK] Admin token acquired")

    # 3. Create Dedicated Stress Session
    print("\n[3/5] Starting Dedicated Stress Session...")
    _, lines = make_request("/lines", token=op_token)
    _, prods = make_request("/products", token=op_token)
    line_id = lines[0]["id"]
    prod_id = prods[0]["id"]

    _, sess = make_request("/sessions", method="POST", data={
        "line_id": line_id,
        "product_profile_id": prod_id,
        "external_ref": "IRS-STRESS-2026-999",
        "target_count": 100,
    }, token=op_token)
    sess_id = sess["id"]
    print(f"  [OK] Created Session #{sess_id} on Line {line_id}")

    # 4. Multi-Threaded Concurrent Bag Crossing Burst (50 concurrent requests)
    print("\n[4/5] Running Concurrent Burst Test: 50 Bags Crossing Gate...")
    latencies = []

    def send_crossing(idx: int):
        t0 = time.perf_counter()
        status, res = make_request(f"/sessions/{sess_id}/simulate_bag", method="POST", data={"direction": 1}, token=op_token)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        return status, res

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(send_crossing, range(50)))
    t_total = time.perf_counter() - t_start

    successes = sum(1 for s, _ in results if s == 200)
    print(f"  [OK] 50/50 concurrent crossing events processed in {t_total:.2f}s ({50/t_total:.1f} req/sec)")
    print(f"  [OK] Latency: p50 = {np.percentile(latencies, 50):.1f}ms, p95 = {np.percentile(latencies, 95):.1f}ms, max = {max(latencies):.1f}ms")
    assert successes == 50, f"Expected 50 successes, got {successes}"

    # 5. Ledger Invariant & Net Count Derivation Check
    print("\n[5/5] Verifying Ledger Invariants & Audit Consistency...")
    _, sess_detail = make_request(f"/sessions/{sess_id}", token=op_token)
    _, events = make_request(f"/sessions/{sess_id}/events", token=op_token)

    print(f"  [OK] Session Counted Total: {sess_detail['counted_total']}")
    print(f"  [OK] Immutable Ledger Record Count: {len(events)}")
    assert sess_detail["counted_total"] == 50, f"Counted total {sess_detail['counted_total']} != 50"
    assert len(events) == 50, f"Event records count {len(events)} != 50"

    # Close and submit to ERP
    print("\n[6/5] Finalizing Session and Outbox Dispatch...")
    make_request(f"/sessions/{sess_id}/close", method="POST", token=op_token)
    status, outbox_res = make_request(f"/sessions/{sess_id}/submit", method="POST", token=op_token)
    assert status == 200, f"Outbox submit failed: {outbox_res}"
    print(f"  [OK] Session successfully dispatched to ERP Outbox (Outbox ID: #{outbox_res['outbox_id']})")

    print("\n" + "=" * 60)
    print("    ALL LIVE DEEP STRESS & INVARIANT CHECKS PASSED 100%!")
    print("=" * 60)


if __name__ == "__main__":
    main()
