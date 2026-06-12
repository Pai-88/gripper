"""Live telemetry dashboard: a FastAPI + WebSocket page showing state, joint
angles, grip, glass-to-servo latency (p50/p95), bus voltage and fault flags.
Far less work than RViz and a much better demo/CV artefact.

Heavy deps (fastapi, uvicorn) are imported lazily so the rest of the stack
imports without them.

    python -m gripper.telemetry.dashboard          # standalone demo
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

# Fault-flag bit -> label, mirrors gripper/protocol.py.
_FLAGS = [(1 << 0, "UNDERVOLT"), (1 << 1, "CLAMP"), (1 << 2, "WATCHDOG"),
          (1 << 3, "OVERTEMP"), (1 << 4, "OVERCURRENT")]


def decode_flags(flags: int) -> list[str]:
    return [name for bit, name in _FLAGS if flags & bit]


def build_app(state_provider: Callable[[], dict[str, Any]]):
    """Create the FastAPI app. ``state_provider`` returns the latest snapshot."""
    from fastapi import FastAPI, WebSocket  # lazy
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="gripper dashboard")

    @app.get("/")
    async def index() -> HTMLResponse:  # noqa: D401
        return HTMLResponse(_INDEX_HTML)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                snap = dict(state_provider())
                snap["fault_labels"] = decode_flags(int(snap.get("flags", 0)))
                await socket.send_text(json.dumps(snap))
                await asyncio.sleep(0.1)  # ~10 Hz UI
        except Exception:  # client disconnect
            return

    return app


async def serve(state_provider: Callable[[], dict[str, Any]],
                host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the dashboard inside the asyncio loop (awaitable)."""
    import uvicorn  # lazy
    config = uvicorn.Config(build_app(state_provider), host=host, port=port,
                            log_level="warning")
    await uvicorn.Server(config).serve()


# Joint display ranges (deg) just for bar scaling in the UI.
_INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>gripper</title><style>
 body{font:14px ui-monospace,monospace;background:#0d0d10;color:#e8e8ea;margin:0;padding:1.2rem}
 h1{font-size:1rem;color:#8ab4f8;margin:0 0 .8rem}
 #state{font-size:1.6rem;font-weight:700;padding:.3rem .8rem;border-radius:.4rem;display:inline-block}
 .row{display:flex;align-items:center;gap:.6rem;margin:.35rem 0}
 .lbl{width:5.5rem;color:#9aa0a6}.bar{flex:1;height:14px;background:#1c1d22;border-radius:7px;overflow:hidden}
 .fill{height:100%;background:#8ab4f8;width:50%}.val{width:5rem;text-align:right}
 .pill{padding:.15rem .5rem;border-radius:.3rem;background:#3c1414;color:#ff8a8a;margin-right:.4rem}
 #lat{margin-top:.8rem;color:#9aa0a6}#flags{margin-top:.6rem;min-height:1.5rem}
 .estop{background:#5c1010;color:#fff}.ok{background:#13361b;color:#9be29b}
</style></head><body>
<h1>gripper · live telemetry</h1>
<div id=state class=ok>—</div>
<div id=joints style=margin-top:1rem></div>
<div id=lat></div><div id=flags></div>
<script>
const NAMES=["base","shoulder","elbow","wrist"], RANGE={base:90,shoulder:90,elbow:120,wrist:90};
const J=document.getElementById('joints');
J.innerHTML=NAMES.map(n=>`<div class=row><span class=lbl>${n}</span>
 <div class=bar><div class=fill id="f_${n}"></div></div><span class=val id="v_${n}">·</span></div>`).join('')
 +`<div class=row><span class=lbl>grip</span><div class=bar><div class=fill id=f_grip style=background:#f8b88a></div></div><span class=val id=v_grip>·</span></div>`;
const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=e=>{const d=JSON.parse(e.data);
 const s=document.getElementById('state');s.textContent=d.state;
 s.className=d.state==='ESTOP'?'estop':'ok';
 for(const n of NAMES){const a=d.joints[n], r=RANGE[n];
  document.getElementById('f_'+n).style.width=(50+50*a/r)+'%';
  document.getElementById('v_'+n).textContent=a+'°';}
 document.getElementById('f_grip').style.width=(100*d.grip)+'%';
 document.getElementById('v_grip').textContent=(d.grip*100|0)+'%';
 document.getElementById('lat').textContent=`latency  p50 ${d.latency_ms.p50} ms   p95 ${d.latency_ms.p95} ms   ·   vbat ${(d.vbat_mV/1000).toFixed(2)} V   ·   seq ${d.seq}`;
 const f=document.getElementById('flags');
 f.innerHTML=(d.fault_labels||[]).map(x=>`<span class=pill>${x}</span>`).join('')||'<span style=color:#5a5>no faults</span>';
};
ws.onclose=()=>document.getElementById('state').textContent='disconnected';
</script></body></html>"""


def main() -> None:
    import math
    import time

    t0 = time.monotonic()

    def demo_snapshot() -> dict:
        t = time.monotonic() - t0
        return {"state": "TELEOP",
                "joints": {"base": round(40 * math.sin(t), 1), "shoulder": 30.0,
                           "elbow": -60.0, "wrist": round(20 * math.sin(t * 2), 1)},
                "grip": round(0.5 + 0.5 * math.sin(t), 3),
                "flags": 0, "vbat_mV": 7400, "seq": int(t * 20),
                "latency_ms": {"p50": 58.0, "p95": 95.0}, "clamped": False}

    asyncio.run(serve(demo_snapshot, port=8000))


if __name__ == "__main__":
    main()
