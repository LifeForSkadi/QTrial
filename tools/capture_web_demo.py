# -*- coding: utf-8 -*-
"""Drive the QTrial web UI via Edge CDP and capture real screenshots.

Usage: python tools/capture_web_demo.py
Outputs: tables/submission/captures/web_filled.png, web_result.png, web_result_full.png
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
OUT = Path(__file__).resolve().parent.parent / "tables" / "submission" / "captures"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8000/?v=3"


def http_json(path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.loads(r.read().decode())


async def cdp(ws, msg_id, method, params=None):
    await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
    while True:
        data = json.loads(await ws.recv())
        if data.get("id") == msg_id:
            return data


async def main():
    proc = subprocess.Popen([
        EDGE, "--headless=new", "--disable-gpu", "--no-first-run",
        f"--remote-debugging-port={PORT}",
        "--user-data-dir=" + str(Path.home() / ".qtrial_edge_profile"),
        "--window-size=1920,1080", "--hide-scrollbars", URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait for devtools
    for _ in range(60):
        try:
            targets = http_json("/json/list")
            if targets:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError("Edge devtools not ready")
    page = next(t for t in targets if t.get("type") == "page")
    ws_url = page["webSocketDebuggerUrl"]

    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        mid = 0

        async def call(method, params=None):
            nonlocal mid
            mid += 1
            return await cdp(ws, mid, method, params)

        await call("Page.enable")
        await call("Runtime.enable")
        await call("Emulation.setDeviceMetricsOverride",
                   {"width": 1920, "height": 1080, "deviceScaleFactor": 2, "mobile": False})
        await call("Page.navigate", {"url": URL})
        await asyncio.sleep(3.0)
        # dismiss possible first-run
        await call("Runtime.evaluate", {"expression": "document.body.click()"})
        await asyncio.sleep(0.5)

        async def shot(path, full=False):
            params = {"format": "png", "captureBeyondViewport": full}
            r = await call("Page.captureScreenshot", params)
            Path(path).write_bytes(base64.b64decode(r["result"]["data"]))

        # 1) fill sample QFT-5
        await call("Runtime.evaluate", {"expression": "document.getElementById('sampleBtn').click()"})
        await asyncio.sleep(1.2)
        await shot(OUT / "web_filled.png")

        # 2) run mapping (turn off qiskit comparison: demo shows QTrial-only metrics)
        await call("Runtime.evaluate", {
            "expression": "document.getElementById('compare').checked=false; "
                          "document.getElementById('runBtn').click(); 'clicked'"})
        # poll for result panel
        ok = False
        for i in range(240):
            await asyncio.sleep(1.0)
            r = await call("Runtime.evaluate", {"expression":
                "(() => {const p=document.getElementById('resultPanel');"
                "const s=document.getElementById('status');"
                "return JSON.stringify({shown: p && p.style.display!=='none',"
                "status: s ? s.innerText.slice(0,200) : ''});})()"})
            val = r["result"].get("result", {}).get("value", "{}")
            st = json.loads(val)
            if i % 5 == 0:
                print("poll", i, st.get("status", "")[:80])
            if st.get("shown"):
                ok = True
                break
        print("result shown:", ok)
        await asyncio.sleep(2.0)
        await shot(OUT / "web_result.png")
        await shot(OUT / "web_result_full.png", full=True)

        # 3) scroll to results for a second framing
        await call("Runtime.evaluate", {
            "expression": "document.getElementById('resultPanel').scrollIntoView({block:'start'}); 1"})
        await asyncio.sleep(1.0)
        await shot(OUT / "web_result_scrolled.png")

    proc.terminate()
    print("captures written to", OUT)


if __name__ == "__main__":
    asyncio.run(main())
