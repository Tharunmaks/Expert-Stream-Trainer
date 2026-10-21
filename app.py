#!/usr/bin/env python3
"""Expert Stream Trainer: Colab-like notebook; each run is a sandboxed CPU subprocess with streamed logs.
Shares accounts, Expert One subscription status and the model library with the main site through es_common.
Anything a job saves to ./model/ (es-moe-v1 format, config.json) is registered in the shared library on exit,
so it shows up in the site's Run page and in the MCP server's list_models/run_moe.  Port 8781."""
import json, os, resource, shutil, signal, subprocess, sys, tempfile, threading, time, uuid
from flask import Flask, Response, abort, jsonify, redirect, render_template, request
import es_common as C

app = Flask(__name__)
SITE_URL = os.environ.get("SITE_URL", "http://127.0.0.1:8780")
CPU_HOURS = {None: 0.5, "one-starter": 4, "one-mid": 12, "one-top": 24}
JOBS = {}
app.jinja_env.globals.update(site_url=SITE_URL)

def me():
    u = C.user_from_request(request)
    if u: u["plan"] = C.subscription(u["id"]); u["limits"] = dict(C.PLAN_LIMITS[u["plan"]], cpu_hours=CPU_HOURS[u["plan"]])
    return u

@app.context_processor
def _ctx(): u = me(); return dict(user=u, sub=u and u["plan"])

@app.get("/")
def notebook():
    u = me()
    if not u: return redirect(f"{SITE_URL}/login?next=" + request.url)
    return render_template("notebook.html", limits=u["limits"], models=C.list_models(u["id"]))

def _limits(cpu_s, mem):
    def f():
        os.setsid(); os.nice(10)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s)); resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 << 30, 2 << 30))
    return f

def _finish(j, u):
    cfgp = os.path.join(j["wd"], "model", "config.json")
    if not os.path.exists(cfgp): return
    cfg = json.load(open(cfgp)); lim = u["limits"]
    if cfg.get("total_params", 0) > lim["max_params"]:
        j["lines"].append(f"[model not saved: {cfg['total_params']:,.0f} params is above your plan's {lim['max_params']:,.0f}; upgrade to Expert One]"); return
    name = j["name"] or f"nb-{j['id']}"; dest = os.path.join(C.MODELS_DIR, f"{u['id'][:8]}-{name}-{j['id']}")
    shutil.move(os.path.join(j["wd"], "model"), dest)
    mid = C.register_model(u["id"], name, dest, cfg, "trainer")
    j["lines"].append(f"[saved model '{name}' to the shared library, id {mid}; it is now in Run and MCP list_models]")

@app.post("/api/run")
def run():
    u = me() or abort(401); d = request.json or {}; cells = d.get("cells", [])
    if not cells or sum(map(len, cells)) > 200_000: abort(400)
    live = [j for j in JOBS.values() if j["uid"] == u["id"] and not j["done"]]
    if len(live) >= u["limits"]["jobs"]: return jsonify(error=f"your plan allows {u['limits']['jobs']} running job(s)"), 429
    jid = uuid.uuid4().hex[:10]; wd = tempfile.mkdtemp(prefix="esnb-")
    src = "".join(f"print('--- cell {i+1} ---', flush=True)\n{c}\n" for i, c in enumerate(cells))
    open(os.path.join(wd, "main.py"), "w").write(src)
    env = dict(PATH="/usr/bin:/bin", HOME=wd, PYTHONUSERBASE=os.path.expanduser("~/.local"), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", ES_PLAN_MAX_PARAMS=str(u["limits"]["max_params"]),
               ES_PLAN_MAX_STEPS=str(u["limits"]["max_steps"]))
    if os.environ.get("ES_BIN_DIR"): env["ES_BIN_DIR"] = os.environ["ES_BIN_DIR"]
    pr = subprocess.Popen([sys.executable, "main.py"], cwd=wd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          preexec_fn=_limits(int(u["limits"]["cpu_hours"] * 3600), 3 << 30), text=True, bufsize=1)
    j = JOBS[jid] = dict(id=jid, uid=u["id"], name=(d.get("name") or "").strip()[:40].replace("/", "-"), proc=pr, lines=[], done=False, started=time.time(), wd=wd)
    def pump():
        for ln in pr.stdout: j["lines"].append(ln.rstrip("\n"))
        pr.wait(); j["lines"].append(f"[exit code {pr.returncode} after {time.time()-j['started']:.1f}s]")
        if pr.returncode == 0:
            try: _finish(j, u)
            except Exception as e: j["lines"].append(f"[could not save model: {e}]")
        j["done"] = True
    threading.Thread(target=pump, daemon=True).start()
    return jsonify(job_id=jid)

@app.get("/api/logs/<jid>")
def logs(jid):
    u = me() or abort(401); j = JOBS.get(jid) or abort(404)
    if j["uid"] != u["id"]: abort(403)
    def gen():
        i = 0
        while True:
            while i < len(j["lines"]): yield f"data: {json.dumps(j['lines'][i])}\n\n"; i += 1
            if j["done"]: yield "event: done\ndata: {}\n\n"; return
            time.sleep(0.3)
    return Response(gen(), mimetype="text/event-stream")

@app.post("/api/stop/<jid>")
def stop(jid):
    u = me() or abort(401); j = JOBS.get(jid) or abort(404)
    if j["uid"] != u["id"]: abort(403)
    try: os.killpg(j["proc"].pid, signal.SIGTERM)
    except OSError: pass
    return jsonify(stopped=True)

if __name__ == "__main__": C.db(); app.run("0.0.0.0", int(os.environ.get("PORT", 8781)), threaded=True)
