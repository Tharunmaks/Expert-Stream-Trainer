# Expert Stream Trainer
Colab-like CPU notebook for Expert Stream. Write Python cells; Run sends them to one sandboxed CPU job
(nice 10, own temp dir, clean env, 2 threads, CPU-time cap set by plan, 3 GB memory cap). Logs stream live; Stop kills the job.

## How it connects to Expert Stream ([Tharunmaks/Stream-](https://github.com/Tharunmaks/Stream-))
Both apps use the shared library `es-common` (folder `es-common/` in Stream-):
- **Shared database**: set the same `DATABASE_URL` on both apps (Postgres `postgresql://...`; SQLite `sqlite:///...` for local dev). Both run the same migrations on start.
- **One login**: set the same `ES_JWT_SECRET` on both. The site issues an `es_session` JWT cookie; set `ES_COOKIE_DOMAIN` (e.g. `.yourdomain`) so both subdomains see it.
- **Plans**: subscriptions (Expert Stream free Starter, Expert One Starter/Mid/Top) are written only by the site's Razorpay flow. The Trainer reads them to set job count, CPU hours and max model size, and its Upgrade links go to the site's Plans page (`SITE_URL`).
- **Model library**: anything a job saves to `./model/` (es-moe-v1 format with `config.json`, e.g. via `es_common.esmoe.train_torch("model", ...)`) is moved to `ES_MODELS_DIR` and registered in the shared `models` table, so it appears in the site's Run page and in the MCP server's `list_models` / `run_moe`. Use the same `ES_MODELS_DIR` on both.
- **Engine**: prebuilt binaries only, found through `ES_BIN_DIR`. No engine source lives here.

## Run
    pip install "git+https://github.com/Tharunmaks/Stream-.git#subdirectory=es-common" flask torch
    export DATABASE_URL=... ES_JWT_SECRET=... ES_MODELS_DIR=... SITE_URL=http://127.0.0.1:8780
    python3 app.py   # port 8781
