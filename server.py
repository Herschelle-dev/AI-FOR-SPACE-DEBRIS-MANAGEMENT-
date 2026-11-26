"""
OrbitGuard backend - ONE file, no changes needed to your HTML.

What it does
  * Serves your existing OrbitGuard HTML page at http://localhost:8000
  * Downloads REAL orbit data (TLEs) from CelesTrak: space stations + known debris clouds
  * Propagates them with SGP4 for the next 3 hours and finds the closest approaches
  * Injects a new "REAL ORBIT CONJUNCTIONS" panel into your page automatically

Setup:   pip install sgp4 numpy
Run:     python server.py
Open:    http://localhost:8000      (do NOT open the HTML file directly)

Put this file next to your HTML (or one folder above a 'frontend' folder containing it).
"""
import glob, json, os, threading, time, urllib.request
from datetime import datetime, timezone, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

BASE = os.path.dirname(os.path.abspath(__file__))
SAT_GROUPS = ["stations"]                                              # the "protected" satellites
DEB_GROUPS = ["cosmos-1408-debris", "fengyun-1c-debris", "iridium-33-debris"]
PER_GROUP = 100        # debris objects used per group (keeps it fast)
HOURS, STEP = 3, 10    # look-ahead window and coarse time step (seconds)
CACHE_HOURS = 6        # re-download TLEs at most every 6 h (CelesTrak asks for this)
RE = 6371.0


# ---------------------------------------------------------------- TLE loading
def load_group(group):
    path = os.path.join(BASE, f"tle_{group}.txt")
    stale = not os.path.exists(path) or time.time() - os.path.getmtime(path) > CACHE_HOURS * 3600
    if stale:
        try:
            url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
            req = urllib.request.Request(url, headers={"User-Agent": "orbitguard-prototype"})
            data = urllib.request.urlopen(req, timeout=20).read().decode()
            if not data.lstrip().startswith(("0 ", "1 ")) and "\n1 " not in data:
                raise ValueError(data[:80])
            open(path, "w").write(data)
        except Exception as e:
            if not os.path.exists(path):
                raise RuntimeError(f"Could not download '{group}' TLEs ({e}). Check your internet connection.")
            print(f"[warn] using cached {group} TLEs ({e})")
    lines = [l.strip() for l in open(path).read().splitlines() if l.strip()]
    out = []
    for i in range(0, len(lines) - 2, 3):
        try:
            out.append((lines[i].lstrip("0 ").strip(), Satrec.twoline2rv(lines[i + 1], lines[i + 2])))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- conjunction screening
def compute():
    sats = [s for g in SAT_GROUPS for s in load_group(g)]
    debs = [d for g in DEB_GROUPS for d in load_group(g)[:PER_GROUP]]
    if not sats or not debs:
        raise RuntimeError("No orbit data loaded.")

    now = datetime.now(timezone.utc)
    jd0, fr0 = jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond / 1e6)
    n = HOURS * 3600 // STEP
    jd = np.full(n, jd0)
    fr = fr0 + np.arange(n) * STEP / 86400.0

    e1, r1, _ = SatrecArray([s for _, s in sats]).sgp4(jd, fr)
    e2, r2, _ = SatrecArray([d for _, d in debs]).sgp4(jd, fr)
    r1[e1 != 0] = np.nan
    r2[e2 != 0] = np.nan

    # coarse scan: for each debris, closest satellite and time
    best_d = np.full(len(debs), np.inf)
    best_s = np.zeros(len(debs), int)
    best_t = np.zeros(len(debs), int)
    for i in range(len(sats)):
        d = np.linalg.norm(r2 - r1[i][None], axis=2)          # (debris, time)
        d = np.where(np.isnan(d), np.inf, d)
        k = d.argmin(axis=1)
        m = d[np.arange(len(debs)), k]
        better = m < best_d
        best_d[better], best_s[better], best_t[better] = m[better], i, k[better]

    results = []
    for j in np.argsort(best_d)[:12]:
        if not np.isfinite(best_d[j]):
            continue
        sat, deb = sats[best_s[j]][1], debs[j][1]
        t0 = best_t[j] * STEP
        ts = np.arange(max(0, t0 - STEP), t0 + STEP + 0.01, 0.25)   # fine refinement around the minimum
        frs = fr0 + ts / 86400.0
        _, ra, va = sat.sgp4_array(np.full(len(ts), jd0), frs)
        _, rb, vb = deb.sgp4_array(np.full(len(ts), jd0), frs)
        d = np.linalg.norm(ra - rb, axis=1)
        if np.isnan(d).all():
            continue
        k = int(np.nanargmin(d))
        miss = float(d[k])
        prob = 5e-3 * np.exp(-miss / 4.0)                           # toy probability model
        results.append({
            "debris": debs[j][0], "satellite": sats[best_s[j]][0],
            "miss_km": round(miss, 2), "tca_s": float(ts[k]),
            "tca_utc": (now + timedelta(seconds=float(ts[k]))).strftime("%H:%M:%S UTC"),
            "rel_speed": round(float(np.linalg.norm(va[k] - vb[k])), 2),
            "alt_km": round(float(np.linalg.norm(rb[k]) - RE)),
            "prob": f"{prob:.1e}",
            "level": "critical" if miss < 5 else "warning" if miss < 25 else "safe",
        })
    results.sort(key=lambda x: x["miss_km"])
    return {"generated": now.strftime("%Y-%m-%d %H:%M:%S UTC"), "n_sats": len(sats),
            "n_debris": len(debs), "hours": HOURS, "results": results}


# ---------------------------------------------------------------- injected UI panel
PANEL = """
<div class="wrap" style="padding-top:0"><div class="panel">
  <div class="ph"><h3>REAL ORBIT CONJUNCTIONS (SGP4 + CelesTrak TLEs)</h3>
    <div class="row"><span id="rcInfo">loading...</span><button class="primary" onclick="loadReal()">REFRESH</button></div></div>
  <div class="tw"><table><thead><tr><th>Debris</th><th>Satellite</th><th>Miss distance</th><th>Closest approach</th>
    <th>Rel. speed</th><th>Debris alt.</th><th>Probability</th><th>Risk</th></tr></thead><tbody id="rcBody"></tbody></table></div>
</div></div>
<script>
async function loadReal(){
  const info=document.getElementById('rcInfo'), body=document.getElementById('rcBody');
  info.textContent='computing real orbits... (first run can take ~10 s)';
  try{
    const r=await fetch('/api/conjunctions'); const j=await r.json();
    if(j.error){info.textContent='Error: '+j.error; return;}
    info.textContent=j.n_sats+' satellites vs '+j.n_debris+' debris, next '+j.hours+' h - '+j.generated;
    body.innerHTML=j.results.map(x=>`<tr><td class="blue"><b>${x.debris}</b></td><td>${x.satellite}</td>
      <td>${x.miss_km} km</td><td>${x.tca_utc}<br><small style="color:#758399">in ${Math.floor(x.tca_s/60)} min</small></td>
      <td>${x.rel_speed} km/s</td><td>${x.alt_km} km</td><td>${x.prob}</td>
      <td><span class="tag ${x.level}">${x.level.toUpperCase()}</span></td></tr>`).join('')
      ||'<tr><td colspan="8">No results</td></tr>';
  }catch(e){info.textContent='Could not reach server - run python server.py and open http://localhost:8000';}
}
loadReal();
</script>
"""


# ---------------------------------------------------------------- tiny web server
def find_html():
    for pattern in ("frontend/*.html", "*.html", "../frontend/*.html"):
        for f in sorted(glob.glob(os.path.join(BASE, pattern))):
            if "ORBITGUARD" in open(f, encoding="utf-8", errors="ignore").read():
                return f
    raise FileNotFoundError("Put your OrbitGuard HTML file next to server.py (or in a 'frontend' folder).")


cache = {"t": 0, "data": None}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            html = open(find_html(), encoding="utf-8").read()
            marker = "<footer>" if "<footer>" in html else "</body>"
            self.send(200, html.replace(marker, PANEL + marker, 1), "text/html")
        elif path == "/api/conjunctions":
            try:
                with lock:
                    if time.time() - cache["t"] > 600 or cache["data"] is None:
                        cache["data"], cache["t"] = compute(), time.time()
                    data = cache["data"]
                self.send(200, json.dumps(data), "application/json")
            except Exception as e:
                self.send(200, json.dumps({"error": str(e)}), "application/json")
        else:
            self.send(404, "Not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("OrbitGuard running ->  http://localhost:8000   (Ctrl+C to stop)")
    ThreadingHTTPServer(("localhost", 8000), Handler).serve_forever()
