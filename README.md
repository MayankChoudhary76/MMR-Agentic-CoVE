# MMR‑Agentic‑CoVE — Multimodal Recommender (Agentic + FAISS)

A compact, agent‑orchestrated recommender system for the Amazon “Beauty” subset:
- Text, image, and lightweight metadata embeddings
- Fusion (concat or weighted sum)
- FAISS ANN index for fast serving
- FastAPI backend + Gradio frontend
- Optional deployment to Hugging Face Spaces (thin UI + API)

---

## Repo layout (key bits)

src/
agents/              # Orchestrator + agents
data/                # Data loader/registry
models/              # Encoders + fusion
service/             # recommend_for_user API surface
utils/               # paths, config utils
scripts/
join_meta.py         # build joined + items_with_meta
build_text_emb.py    # item/user text embeddings (SBERT)
build_image_emb.py   # item image embeddings (CLIP)
build_meta_emb.py    # hashed/bucket meta embeddings
build_faiss.py       # build FAISS index over fused vectors
run_agent.py         # CLI: prepare / index
api/
app_api.py           # FastAPI app (POST /recommend, /healthz)
ui/
app.py               # Gradio UI (calls API)
data/
raw/                 # raw reviews/meta
processed/           # parquet + embeddings + FAISS
logs/                  # metrics & plots

---

bash
## Quickstart (local)

> **Prereqs:** Python 3.10+; (optional) GPU w/ CUDA for faster image/text models.

```bash
# 1) clone and enter
git clone https://github.com/MayankChoudhary76/MMR-Agentic-CoVE.git
cd MMR-Agentic-CoVE

# 2) create & activate venv
python -m venv .venv
source .venv/bin/activate

# 3) install deps
pip install -U pip wheel
pip install -r requirements.txt   # (if present)
# or, if your repo splits UI/API deps:
# pip install -r requirements-api.txt
# pip install -r requirements-ui.txt

# 4) set module path
export PYTHONPATH=$(pwd)

Prepare data (agent will run join → text → image → meta)
bash
PYTHONPATH=$(pwd) python -m scripts.run_agent --intent prepare --dataset beauty
output
You should see:
	•	data/processed/beauty/joined.parquet
	•	item_text_emb.parquet, item_image_emb.parquet, item_meta_emb.parquet
	•	quick text baseline metrics in logs/*.json

Build FAISS index
bash
PYTHONPATH=$(pwd) python -m scripts.run_agent --intent index \
  --dataset beauty \
  --fusion concat --w_text 1 --w_image 1 --w_meta 0.4 \
  --faiss_name beauty_concat_best
 output 
 data/processed/beauty/index/
  items_beauty_concat_best.faiss
  items_beauty_concat_best.npy
  
 Run the API
 bash
 # ensure processed data + FAISS exist
PYTHONPATH=$(pwd) uvicorn api.app_api:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/docs
Sanity test:
bash
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"beauty","user_id":"A3CIUOJXQ5VDQ2","k":10,
       "fusion":"concat","w_text":1,"w_image":1,"w_meta":0.4,
       "use_faiss":true,"faiss_name":"beauty_concat_best","exclude_seen":true}'
       
Run the UI (local Gradio)
bash
# calls the FastAPI; set this to your local or Space API URL
export API_URL="http://localhost:8000"
python ui/app.py
# Gradio will print a local URL

Hugging Face Spaces (thin UI + API)

Option A (recommended): 2 Spaces
	•	API Space: FastAPI app (app.py with FastAPI + requirements.txt)
	•	UI Space: Gradio app that calls the API URL

Use Spaces for serving; use Paperspace/local only for heavy prepare + index (then copy the built data/processed/.../index to the API Space repo if you want it to serve without regenerating).

API Tips
	•	Keep only what you need: app.py, requirements.txt, and any small artifacts (index files for the demo dataset).
	•	Your app.py should import recommend_for_user from src/service/recommender or bundle only the minimal service code you need.

UI Tips
	•	In ui/app.py, set API_URL to your API Space:
bash
API_URL = os.environ.get("API_URL", "https://<your-username>-cove-api.hf.space")

Makefile (handy commands)

See the Makefile in this repo; it wraps common flows:
	•	make venv install
	•	make prepare
	•	make index
	•	make api
	•	make ui
	•	make test-api
    
Troubleshooting
	•	Module not found (src.*)
Ensure export PYTHONPATH=$(pwd) or run commands as PYTHONPATH=$(pwd) python ....
	•	Missing FAISS file when calling API
Build index (scripts/build_faiss.py) and set use_faiss=true, faiss_name to the saved name.
	•	Large model downloads
On CPU, expect SBERT/CLIP to download once. Use GPU for speed; on Spaces, prefer prebuilt artifacts.

License

MIT (see LICENSE in the repo)