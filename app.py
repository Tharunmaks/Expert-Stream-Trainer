#!/usr/bin/env python3
"""Expert Stream Trainer: Colab-like notebook; each run is a sandboxed CPU subprocess with streamed logs.
Shares accounts, Expert One subscription status and the model library with the main site through es_common.
Anything a job saves to ./model/ (es-moe-v1 format, config.json) is registered in the shared library on exit,
so it shows up in the site's Run page and in the MCP server's list_models/run_moe.  Port 8781."""
import json, os, resource, shutil, signal, subprocess, sys, tempfile, threading, time, uuid
from flask import Flask, Response, abort, jsonify, redirect, render_template, request
import es_common as C

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 70 << 20
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

# ---------------- Train MoE: sparse-backprop trainer (prebuilt es_train moe --sparse [--p6]) ----------------
# Frozen packed-int4 experts streamed from disk; trains LoRA r4 on routed experts + routers + embedding + head.
# Plan limits apply to ACTIVE params per token (what each step computes): free 2.5M, Starter 20M, Mid 100M, Top 1B.
ES_HOME = os.path.expanduser("~/.expertstream")
CORPUS = os.environ.get("ES_CORPUS", os.path.join(ES_HOME, "data", "gutenberg.txt"))   # public-domain Project Gutenberg books
EXPERTS_DIR = os.environ.get("ES_EXPERTS_DIR", os.path.join(ES_HOME, "experts"))
LAB_EXPERTS = os.environ.get("ES_FROZEN_10B", "/home/box/research/expert-stream-lab/p1/experts.bin")  # same file moe10b-gut reuses
MOE_PRESETS = {
    "tiny":  dict(label="Tiny · 2.1M", layers=2,  experts=4,  dim=512,  ffn=256,  batch=8,  file="tiny-2m.bin"),
    "small": dict(label="Small · 0.27B", layers=8, experts=32, dim=1024, ffn=512, batch=16, file="small-0.27b.bin"),
    "2b":    dict(label="2B", layers=16, experts=32, dim=2048, ffn=1024, batch=16, file="moe-2.1b.bin"),
    "10b":   dict(label="10B (shared frozen file)", layers=16, experts=76, dim=2048, ffn=2048, batch=16, file=LAB_EXPERTS),
}
for p in MOE_PRESETS.values():
    p["total"] = p["layers"] * p["experts"] * 2 * p["dim"] * p["ffn"]; p["active"] = p["layers"] * 2 * p["dim"] * p["ffn"]
    p["path"] = p["file"] if os.path.isabs(p["file"]) else os.path.join(EXPERTS_DIR, p["file"])

@app.get("/moe")
def moe_page():
    u = me()
    if not u: return redirect(f"{SITE_URL}/login?next=" + request.url)
    return render_template("moe.html", limits=u["limits"], presets=MOE_PRESETS, corpus=os.path.exists(CORPUS),
                           models=[m for m in C.list_models(u["id"]) if m.get("source") == "trainer-moe"])

def _moe_limits(cpu_s):
    def f():
        os.setsid(); os.nice(10); resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 30, 8 << 30))   # no RLIMIT_AS: the frozen expert file is mmap'd
    return f

@app.post("/api/moe/start")
def moe_start():
    u = me() or abort(401); f = request.form; lim = u["limits"]
    p = MOE_PRESETS.get(f.get("preset", "")) or abort(400)
    if p["active"] > lim["max_params"]:
        return jsonify(error=f"{p['label']} computes {p['active']/1e6:.1f}M active params per token; your plan allows {lim['max_params']/1e6:g}M. Upgrade to Expert One."), 403
    live = [j for j in JOBS.values() if j["uid"] == u["id"] and not j["done"]]
    if len(live) >= lim["jobs"]: return jsonify(error=f"your plan allows {lim['jobs']} running job(s)"), 429
    steps = max(1, min(int(f.get("steps", 100)), int(lim["max_steps"]))); threads = max(1, min(int(f.get("threads", 2)), 4))
    mode = "plain" if f.get("mode") == "plain" else "p6"
    if p["file"] == LAB_EXPERTS and not os.path.exists(LAB_EXPERTS): return jsonify(error="the shared 10B frozen expert file is not on this server"), 503
    jid = uuid.uuid4().hex[:10]; wd = tempfile.mkdtemp(prefix="esmoe-"); data = os.path.join(wd, "data.txt")
    up = request.files.get("file"); txt = f.get("text", "")
    if up and up.filename: up.save(data); src = "upload:" + os.path.basename(up.filename)[:60]
    elif txt.strip(): open(data, "w").write(txt); src = "pasted text"
    elif os.path.exists(CORPUS): data = CORPUS; src = "built-in Project Gutenberg corpus"
    else: return jsonify(error="no dataset: paste text, upload a file, or install the built-in corpus"), 400
    if os.path.getsize(data) < 8192: return jsonify(error="dataset too small: need at least 8 KB of text (10% is held out)"), 400
    if os.path.getsize(data) > 64 << 20: return jsonify(error="dataset too large (64 MB max)"), 400
    os.makedirs(EXPERTS_DIR, exist_ok=True); os.makedirs(os.path.join(wd, "model"), exist_ok=True)
    ck = os.path.join(wd, "model", "m.ckpt")
    cmd = [C.es_bin("es_train"), "moe", "--sparse"] + (["--p6"] if mode == "p6" else []) + [
        "--layers", str(p["layers"]), "--experts", str(p["experts"]), "--dim", str(p["dim"]), "--ffn", str(p["ffn"]),
        "--batch", str(p["batch"]), "--seq", "128", "--steps", str(steps), "--eval-every", str(max(1, min(25, steps // 4 or 1))),
        "--data", data, "--experts-file", p["path"], "--save", ck]
    env = dict(PATH="/usr/bin:/bin", HOME=wd, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    pr = subprocess.Popen(cmd, cwd=wd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=_moe_limits(int(lim["cpu_hours"] * 3600)), text=True, bufsize=1)
    name = (f.get("name") or "").strip()[:40].replace("/", "-") or f"moe-{f.get('preset')}"
    j = JOBS[jid] = dict(id=jid, uid=u["id"], name=name, proc=pr, lines=[], done=False, started=time.time(), wd=wd, kind="moe")
    j["lines"].append(f"[job {jid}: {p['label']} ({p['total']/1e9:.3g}B total, {p['active']/1e6:.1f}M active), mode {mode}, {steps} steps, {threads} thread(s), nice 10, data: {src}]")
    def pump():
        last, hist = {}, []
        for ln in pr.stdout:
            ln = ln.rstrip("\n"); j["lines"].append(ln); w = ln.split()
            if len(w) > 3 and w[0] == "step" and w[2] == "train_ce": hist.append([int(w[1]), float(w[3])])
            if ln.startswith("RESULT"): last = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
        pr.wait(); j["lines"].append(f"[exit code {pr.returncode} after {time.time()-j['started']:.1f}s]")
        try:
            if os.path.exists(ck):
                cfg = dict(format="es-sparse-v1", ckpt="m.ckpt", experts_file=p["path"], layers=p["layers"], experts=p["experts"], dim=p["dim"], ffn=p["ffn"],
                           topk=1, materialized=True, total_params=p["total"], active_params=p["active"],
                           train=dict(loss_history=hist[-200:] or [[0, 0.0]], mode=mode, steps=steps, threads=threads, data=src, train_tok_s=float(last.get("train_tok/s", 0) or 0),
                                      heldout_before=float(last.get("heldout_before", 0) or 0), heldout_after=float(last.get("heldout_after", 0) or 0)),
                           note=f"Trainer 'Train MoE' job {jid}: frozen random int4 experts + trained LoRA r4, routers, embedding, head.")
                json.dump(cfg, open(os.path.join(wd, "model", "config.json"), "w"), indent=1)
                dest = os.path.join(C.MODELS_DIR, f"{u['id'][:8]}-{name}-{jid}"); shutil.move(os.path.join(wd, "model"), dest)
                mid = C.register_model(u["id"], name, dest, cfg, "trainer-moe")
                j["lines"].append(f"[saved checkpoint and registered '{name}' (id {mid}) in the shared library; open Run on the site to generate]"); j["model_id"] = mid
            else: j["lines"].append("[no checkpoint was written, nothing registered]")
        except Exception as e: j["lines"].append(f"[could not register model: {e}]")
        j["done"] = True
    threading.Thread(target=pump, daemon=True).start()
    return jsonify(job_id=jid)

if __name__ == "__main__": C.db(); app.run("0.0.0.0", int(os.environ.get("PORT", 8781)), threaded=True)
