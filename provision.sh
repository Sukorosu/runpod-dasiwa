#!/usr/bin/env bash
# =============================================================================
#  DaSiWa MiniMax H3 - RunPod Provisioning (ohne eigenes Docker-Image)
#
#  Gedacht fuer das offizielle Image runpod/pytorch:1.4.1-cu1300-torch2130-ubuntu2404
#  Container Start Command (siehe README):
#    bash -c "curl -fsSL --retry 30 --retry-delay 2 --retry-all-errors <RAW-URL>/provision.sh -o /post_start.sh; exec /start.sh"
#
#  /start.sh des RunPod-Images startet nginx, SSH und JupyterLab und ruft danach
#  /post_start.sh (= dieses Skript) auf. Dieses Skript kehrt SOFORT zurueck und
#  erledigt die eigentliche Arbeit im Hintergrund:
#    1. ComfyUI + Manager + KJNodes + DaSiWa-Nodes nach /workspace (nur falls fehlend)
#    2. venv auf dem Volume (nutzt Torch cu130 aus dem Image mit, Fallback: eigenes Torch)
#    3. Modelle + CivitAI-LoRAs parallel mit aria2c (vorhandene werden uebersprungen)
#    4. C-MMH3 Workflow ablegen, ComfyUI auf 0.0.0.0:8188 starten
#  Logs: /workspace/dasiwa/provision.log, /workspace/dasiwa/download.log, /workspace/comfyui.log
# =============================================================================

# ---------------------------------------------------------------------------
# Phase 0: sofort in den Hintergrund, damit /start.sh weiterlaeuft
# ---------------------------------------------------------------------------
if [[ "${1:-}" != "--run" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then WS=/workspace; else WS=/root; fi
  mkdir -p "$WS/dasiwa"
  cp -f "$(readlink -f "$0")" "$WS/dasiwa/provision.sh" 2>/dev/null || true
  echo "[dasiwa] Provisioning laeuft im Hintergrund - Log: $WS/dasiwa/provision.log"
  nohup setsid bash "$WS/dasiwa/provision.sh" --run >> "$WS/dasiwa/provision.log" 2>&1 < /dev/null &
  exit 0
fi

set -uo pipefail

# ---------------------------------------------------------------------------
# Einstellungen (alle per Environment Variable ueberschreibbar)
# ---------------------------------------------------------------------------
if [[ -d /workspace && -w /workspace ]]; then WS=/workspace; else WS=/root; fi
DASIWA_HOME="$WS/dasiwa"
COMFY_DIR="${COMFY_DIR:-$WS/ComfyUI}"
VENV="${VENV_DIR:-$WS/venv-dasiwa}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_REF="${COMFY_REF:-}"                  # leer = neuester Stand, sonst Tag/Commit
START_EARLY="${START_EARLY:-1}"             # 1 = ComfyUI startet, waehrend Modelle noch laden
UPDATE_NODES="${UPDATE_NODES:-0}"           # 1 = git pull fuer ComfyUI + Nodes bei jedem Start
UPDATE_WORKFLOW="${UPDATE_WORKFLOW:-0}"     # 1 = neuesten C-MMH3 von GitHub holen
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
WF_REPO_RAW="https://raw.githubusercontent.com/darksidewalker/dasiwa-comfyui-workflows/main/C-MMH3"
WF_DEFAULT_FILE="DaSiWa MiniMaxH3 MythicAlchemy C-MMH3-33.json"

export MODELS_DIR="$COMFY_DIR/models"
BIN="$DASIWA_HOME/bin"
mkdir -p "$BIN" "$DASIWA_HOME/markers"

log() { echo "[dasiwa] $(date '+%H:%M:%S') $*"; }

exec 9> "$DASIWA_HOME/.lock"
if ! flock -n 9; then log "Provisioning laeuft bereits - Abbruch."; exit 0; fi

log "===================== Provisioning startet ====================="
log "Workspace: $WS | ComfyUI: $COMFY_DIR | venv: $VENV"

# RunPod-Umgebung (Token usw.) auch fuer spaetere Jupyter-Terminals verfuegbar machen
[[ -f /etc/rp_environment ]] && source /etc/rp_environment

# ---------------------------------------------------------------------------
# Hilfsskripte schreiben (werden bei jedem Start aktualisiert)
# ---------------------------------------------------------------------------
cat > "$BIN/resolve_downloads.py" <<'PY_EOF'
#!/usr/bin/env python3
"""Loest LoRA-Eintraege (CivitAI-IDs/URLs, HF-URLs) in eine aria2c-Eingabeliste auf.
Aufruf: resolve_downloads.py <zielordner> <eintraege...>   (Trenner: , ; Zeilenumbruch)
Eintrag optional mit eigenem Dateinamen: <eintrag>|<dateiname>"""
import json, os, re, sys, urllib.parse, urllib.request

API = "https://civitai.com/api/v1"
TOKEN = os.environ.get("CIVITAI_TOKEN", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
EXPECTED_BASE = os.environ.get("EXPECTED_BASE_MODEL", "MiniMax H3")

def log(m): print(f"[loras] {m}", file=sys.stderr)

def api_get(path):
    req = urllib.request.Request(API + path, headers={"User-Agent": "runpod-dasiwa/1.0"})
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def with_token(url):
    if not TOKEN:
        return url
    return f"{url}{'&' if '?' in url else '?'}token={urllib.parse.quote(TOKEN)}"

def safe_name(n):
    n = os.path.basename(n.strip())
    return re.sub(r"[^\w.\-+() \[\]]", "_", n) or "download.safetensors"

def resolve_version(vid, file_id=None):
    v = api_get(f"/model-versions/{vid}")
    files = v.get("files") or []
    f = next((x for x in files if file_id and str(x.get("id")) == str(file_id)), None) \
        or next((x for x in files if x.get("primary")), None) \
        or next((x for x in files if x.get("type") == "Model"), None) \
        or (files[0] if files else None)
    if f is None:
        raise RuntimeError(f"Version {vid} hat keine Dateien")
    base = v.get("baseModel") or "?"
    log(f"Version {vid}: '{(v.get('model') or {}).get('name','?')}' / '{v.get('name')}' (Basis: {base}) -> {f['name']}")
    if EXPECTED_BASE and base != EXPECTED_BASE:
        log(f"  WARNUNG: Basismodell ist '{base}', nicht '{EXPECTED_BASE}' - LoRA passt evtl. nicht.")
    if v.get("trainedWords"):
        log(f"  Trigger-Woerter: {', '.join(v['trainedWords'])}")
    return with_token(f.get("downloadUrl") or f"https://civitai.com/api/download/models/{vid}"), f["name"]

def resolve(entry):
    name = None
    if "|" in entry:
        entry, name = (s.strip() for s in entry.split("|", 1))
    headers = []
    if re.fullmatch(r"\d+", entry):
        url, fname = resolve_version(entry)
    elif "civitai.com" in entry or "civitai.red" in entry:
        u = urllib.parse.urlparse(entry); q = urllib.parse.parse_qs(u.query)
        m_dl = re.search(r"/api/download/models/(\d+)", u.path); m_model = re.search(r"/models/(\d+)", u.path)
        if m_dl:
            url, fname = resolve_version(m_dl.group(1), (q.get("fileId") or [None])[0])
        elif "modelVersionId" in q:
            url, fname = resolve_version(q["modelVersionId"][0])
        elif m_model:
            vs = api_get(f"/models/{m_model.group(1)}").get("modelVersions") or []
            if not vs:
                raise RuntimeError("Modell hat keine Versionen")
            log(f"Modell {m_model.group(1)} ohne Versions-ID -> neueste Version {vs[0]['id']}")
            url, fname = resolve_version(vs[0]["id"])
        else:
            raise RuntimeError(f"CivitAI-URL nicht erkannt: {entry}")
    elif entry.startswith(("http://", "https://")):
        url = entry
        fname = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(entry).path)) or "download.safetensors"
        if "huggingface.co" in entry and HF_TOKEN:
            headers.append(f"Authorization: Bearer {HF_TOKEN}")
    else:
        raise RuntimeError(f"Eintrag nicht erkannt: {entry}")
    return url, safe_name(name or fname), headers

def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr); sys.exit(2)
    target = sys.argv[1]
    entries = [e.strip() for e in re.split(r"[,;\n]+", " ".join(sys.argv[2:])) if e.strip()]
    for e in entries:
        try:
            url, fname, headers = resolve(e)
        except Exception as ex:  # noqa: BLE001
            log(f"FEHLER bei '{e}': {ex}"); continue
        p = os.path.join(target, fname)
        if os.path.isfile(p) and not os.path.exists(p + ".aria2"):
            log(f"vorhanden, uebersprungen: {fname}"); continue
        print(url); print(f"  dir={target}"); print(f"  out={fname}")
        for h in headers:
            print(f"  header={h}")

if __name__ == "__main__":
    main()
PY_EOF

cat > "$BIN/download_models.sh" <<'DL_EOF'
#!/usr/bin/env bash
# Laedt alle Modelle fuer den DaSiWa MiniMax H3 Workflow parallel (aria2c).
# Vorhandene Dateien werden uebersprungen, abgebrochene fortgesetzt.
# Manuell:  CIVITAI_LORAS="123,456" /workspace/dasiwa/bin/download_models.sh
set -uo pipefail
[[ -f /etc/rp_environment ]] && source /etc/rp_environment
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [[ -d /workspace && -w /workspace ]]; then _WS=/workspace; else _WS=/root; fi
MODELS_DIR="${MODELS_DIR:-$_WS/ComfyUI/models}"
DASIWA_VARIANT="${DASIWA_VARIANT:-v4-turbo-int8}"
TEXT_ENCODER="${TEXT_ENCODER:-nvfp4}"
VIDEO_VAE="${VIDEO_VAE:-int8}"
DOWNLOAD_EXTRAS="${DOWNLOAD_EXTRAS:-1}"
FORGE_MODEL="${FORGE_MODEL:-gguf-8b-q8}"
CIVITAI_LORAS="${CIVITAI_LORAS:-}"
DASIWA_EXTRA_VARIANTS="${DASIWA_EXTRA_VARIANTS-v4-int8}"   # leer = keine Zusatz-Variante
# RunPod setzt nicht benutzte Template-Variablen oft auf "false" - wie leer behandeln
_is_off() { case "${1,,}" in false|0|none|no|off) return 0 ;; *) return 1 ;; esac; }
_is_off "$CIVITAI_LORAS" && CIVITAI_LORAS=""
_is_off "$DASIWA_EXTRA_VARIANTS" && DASIWA_EXTRA_VARIANTS=""
EXTRA_DOWNLOADS="${EXTRA_DOWNLOADS:-}"
_is_off "$EXTRA_DOWNLOADS" && EXTRA_DOWNLOADS=""
CIVITAI_TOKEN="${CIVITAI_TOKEN:-${civitai_token:-}}"
HF_TOKEN="${HF_TOKEN:-}"
_is_off "$CIVITAI_TOKEN" && CIVITAI_TOKEN=""
_is_off "$HF_TOKEN" && HF_TOKEN=""
ARIA_CONNECTIONS="${ARIA_CONNECTIONS:-16}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-6}"
SKIP_BASE_MODELS="${SKIP_BASE_MODELS:-0}"   # 1 = nur LoRAs/Extras (fuer das Notebook)
WF_MODEL_DEFAULT="MiniMaxH3/dasiwa_minimax_h3_ref2va_v4_pruned_turbo_int8.safetensors"
HF="https://huggingface.co"; CIVI="https://civitai.com/api/download/models"
log() { echo "[download] $*"; }
LIST="$(mktemp /tmp/aria2-list.XXXXXX)"; trap 'rm -f "$LIST"' EXIT

add_file() {  # <url> <unterordner unter models> <dateiname> [hf|civitai]
  local url="$1" sub="$2" name="$3" auth="${4:-}" dir="$MODELS_DIR/$2"
  mkdir -p "$dir"
  if [[ -s "$dir/$name" && ! -e "$dir/$name.aria2" ]]; then log "vorhanden: $sub/$name"; return 0; fi
  if [[ "$auth" == "civitai" ]]; then
    if [[ -z "$CIVITAI_TOKEN" ]]; then log "FEHLER: $name braucht CIVITAI_TOKEN - uebersprungen."; return 0; fi
    url="${url}$( [[ "$url" == *\?* ]] && echo '&' || echo '?')token=${CIVITAI_TOKEN}"
  fi
  { echo "$url"; echo "  dir=$dir"; echo "  out=$name"
    [[ "$auth" == "hf" && -n "$HF_TOKEN" ]] && echo "  header=Authorization: Bearer $HF_TOKEN"; } >> "$LIST"
  log "geplant:   $sub/$name"
}

if [[ "$SKIP_BASE_MODELS" != "1" ]]; then
  # 1) DaSiWa MiniMax H3 (CivitAI Modell 2877206) - <versionId>:<fileId>
  # dasiwa_map <variante> -> setzt DASIWA_SRC und DASIWA_FILE
  dasiwa_map() {
    DASIWA_FILE=""
    case "$1" in
      v4-turbo-int8)  DASIWA_SRC="3392061:3281858"; DASIWA_FILE="$WF_MODEL_DEFAULT" ;;                                      # 19.5 GB, 4-8 Steps
      v4-turbo-int4)  DASIWA_SRC="3392061:3281790"; DASIWA_FILE="MiniMaxH3/DasiwaMinimaxH3_dasiwaHybridV4Turbo_3281790.safetensors" ;;
      v4-turbo-int4b) DASIWA_SRC="3392061:3281836"; DASIWA_FILE="MiniMaxH3/DasiwaMinimaxH3_dasiwaHybridV4Turbo_3281836.safetensors" ;;
      v4-int8)        DASIWA_SRC="3391862:3281578"; DASIWA_FILE="MiniMaxH3/dasiwa_minimax_h3_hybrid_v4_int8.safetensors" ;;     # 19.5 GB, Non-Turbo, 20-25 Steps
      v4-int4)        DASIWA_SRC="3391862:3281517"; DASIWA_FILE="MiniMaxH3/DasiwaMinimaxH3_dasiwaHybridV4_3281517.safetensors" ;;
      v4-int4b)       DASIWA_SRC="3391862:3281560"; DASIWA_FILE="MiniMaxH3/DasiwaMinimaxH3_dasiwaHybridV4_3281560.safetensors" ;;
      v3-turbo-int8)  DASIWA_SRC="3374439:3263048"; DASIWA_FILE="MiniMaxH3/DasiwaMinimaxH3_dasiwaHybridTurboV3_3263048.safetensors" ;;
      v3-turbo-int4)  DASIWA_SRC="3374439:3262925"; DASIWA_FILE="MiniMaxH3/DasiwaMinimaxH3_dasiwaHybridTurboV3_3262925.safetensors" ;;
      custom)         DASIWA_SRC="custom" ;;
      none|false|0|"") DASIWA_SRC="" ;;
      *) DASIWA_SRC="unknown" ;;
    esac
  }
  dasiwa_map "$DASIWA_VARIANT"
  if [[ "$DASIWA_SRC" == "unknown" ]]; then
    log "Unbekannte DASIWA_VARIANT '$DASIWA_VARIANT' - nehme v4-turbo-int8"; dasiwa_map v4-turbo-int8
  fi
  if [[ "$DASIWA_SRC" == "custom" ]]; then
    if [[ -n "${DASIWA_MODEL_URL:-}" ]]; then
      DASIWA_FILE="MiniMaxH3/${DASIWA_MODEL_NAME:-$(basename "${DASIWA_MODEL_URL%%\?*}")}"
      auth=""; [[ "$DASIWA_MODEL_URL" == *civitai.com* ]] && auth="civitai"; [[ "$DASIWA_MODEL_URL" == *huggingface.co* ]] && auth="hf"
      add_file "$DASIWA_MODEL_URL" "diffusion_models/MiniMaxH3" "$(basename "$DASIWA_FILE")" "$auth"
    else
      log "DASIWA_VARIANT=custom, aber DASIWA_MODEL_URL fehlt."
    fi
  elif [[ -n "$DASIWA_SRC" ]]; then
    add_file "$CIVI/${DASIWA_SRC%%:*}?fileId=${DASIWA_SRC##*:}" "diffusion_models/MiniMaxH3" "$(basename "$DASIWA_FILE")" civitai
  fi
  MAIN_DASIWA_FILE="$DASIWA_FILE"

  # 1b) Zusaetzliche DaSiWa-Varianten (Standard: v4-int8 = Non-Turbo), landen nur im Ordner,
  #     der Workflow bleibt auf der Hauptvariante (Turbo).
  if [[ -n "$DASIWA_EXTRA_VARIANTS" ]]; then
    IFS=', ' read -ra _xv <<< "$DASIWA_EXTRA_VARIANTS"
    for xv in "${_xv[@]}"; do
      [[ -z "$xv" || "$xv" == "$DASIWA_VARIANT" ]] && continue
      dasiwa_map "$xv"
      case "$DASIWA_SRC" in
        ""|custom|unknown) log "DASIWA_EXTRA_VARIANTS: '$xv' unbekannt/nicht erlaubt - uebersprungen." ;;
        *) add_file "$CIVI/${DASIWA_SRC%%:*}?fileId=${DASIWA_SRC##*:}" "diffusion_models/MiniMaxH3" "$(basename "$DASIWA_FILE")" civitai ;;
      esac
    done
  fi
  DASIWA_FILE="$MAIN_DASIWA_FILE"
  echo "$DASIWA_FILE" > "$MODELS_DIR/.dasiwa_selected_model"

  # 2) Text-Encoder Qwen3-VL-32B
  case "$TEXT_ENCODER" in
    int4) TE_FILE="qwen3vl_32b_minimax_h3_int4_convrot.safetensors"; TE_URL="$HF/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/$TE_FILE" ;;
    int8) TE_FILE="qwen3vl_32b_minimax_h3_int8_convrot.safetensors"; TE_URL="$HF/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/$TE_FILE" ;;
    none) TE_FILE=""; TE_URL="" ;;
    *)    TE_FILE="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";   TE_URL="$HF/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/$TE_FILE" ;;
  esac
  [[ -n "$TE_URL" ]] && add_file "$TE_URL" text_encoders "$TE_FILE" hf
  echo "${TE_FILE:-}" > "$MODELS_DIR/.dasiwa_selected_te"

  # 3) VAEs
  if [[ "$VIDEO_VAE" == "int8" || "$VIDEO_VAE" == "both" ]]; then
    add_file "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors" vae/MiniMaxH3 minimax_h3_video_vae_int8_convrot.safetensors hf
  fi
  if [[ "$VIDEO_VAE" == "fp16" || "$VIDEO_VAE" == "both" ]]; then
    add_file "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" vae/MiniMaxH3 minimax_h3_video_vae_fp16.safetensors hf
  fi
  echo "$VIDEO_VAE" > "$MODELS_DIR/.dasiwa_selected_vae"
  add_file "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" vae/MiniMaxH3 minimax_h3_audio_vae_fp32.safetensors hf

  # 4) Extras aus dem Workflow
  if [[ "$DOWNLOAD_EXTRAS" == "1" ]]; then
    add_file "$HF/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors" vae_approx taeh3.safetensors hf
    add_file "$HF/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors" \
             latent_upscale_models minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors hf
    add_file "$HF/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26.safetensors" frame_interpolation rife_v4.26.safetensors hf
    add_file "$HF/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN.safetensors" upscale_models 2x-AnimeSharpV4_RCAN.safetensors hf
  fi

  # 4b) LLM fuer "Prompt Forge" / "H3 Forge" (DaSiWa-Nodes) -> models/llm/
  #     GGUF-Varianten brauchen llama-cpp-python (CUDA-Wheel wird von provision.sh installiert)
  #     und die mmproj-Datei im selben Ordner, damit das Modell Bilder "sehen" kann.
  QG="$HF/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main"
  QG4="$HF/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main"
  case "$FORGE_MODEL" in
    1|true|gguf-8b-q8|"")
      add_file "$QG/Qwen3VL-8B-Instruct-Q8_0.gguf"        llm/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q8_0.gguf hf          # 8.7 GB
      add_file "$QG/mmproj-Qwen3VL-8B-Instruct-F16.gguf"  llm/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-F16.gguf hf ;; # 1.2 GB
    gguf-8b-q4)
      add_file "$QG/Qwen3VL-8B-Instruct-Q4_K_M.gguf"      llm/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf hf        # 5.0 GB
      add_file "$QG/mmproj-Qwen3VL-8B-Instruct-F16.gguf"  llm/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-F16.gguf hf ;;
    gguf-4b-q8)
      add_file "$QG4/Qwen3VL-4B-Instruct-Q8_0.gguf"       llm/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q8_0.gguf hf          # 4.3 GB
      add_file "$QG4/mmproj-Qwen3VL-4B-Instruct-F16.gguf" llm/Qwen3-VL-4B-Instruct-GGUF mmproj-Qwen3VL-4B-Instruct-F16.gguf hf ;; # 0.8 GB
    hf-8b|hf-4b)
      # Hugging-Face-Ordner (Transformers-Backend, kein llama-cpp noetig): 8B = 17.5 GB, 4B = 8.9 GB
      repo="Qwen/Qwen3-VL-8B-Instruct"; [[ "$FORGE_MODEL" == "hf-4b" ]] && repo="Qwen/Qwen3-VL-4B-Instruct"
      while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        d="$(dirname "$f")"; [[ "$d" == "." ]] && d="" || d="/$d"
        add_file "$HF/$repo/resolve/main/$f" "llm/$(basename "$repo")$d" "$(basename "$f")" hf
      done < <(curl -fsSL ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$HF/api/models/$repo/tree/main?recursive=true" \
               | python3 -c 'import json,sys; [print(x["path"]) for x in json.load(sys.stdin) if x.get("type")=="file" and not x["path"].startswith(".")]' 2>/dev/null) ;;
    0|false|none) log "FORGE_MODEL=none - kein LLM fuer Prompt Forge." ;;
    *) log "Unbekanntes FORGE_MODEL '$FORGE_MODEL' - uebersprungen." ;;
  esac
fi

# 5) LoRAs + Extra-Downloads
if [[ -n "$CIVITAI_LORAS" ]]; then
  mkdir -p "$MODELS_DIR/loras/${LORA_SUBDIR:-}"
  CIVITAI_TOKEN="$CIVITAI_TOKEN" HF_TOKEN="$HF_TOKEN" \
    python3 "$SCRIPT_DIR/resolve_downloads.py" "$MODELS_DIR/loras/${LORA_SUBDIR:-}" "$CIVITAI_LORAS" >> "$LIST"
  [[ -z "$CIVITAI_TOKEN" ]] && log "Hinweis: kein CIVITAI_TOKEN - viele CivitAI-LoRAs lassen sich ohne Token nicht laden."
fi
if [[ -n "$EXTRA_DOWNLOADS" ]]; then   # "<url>|<ordner unter models>[|dateiname]" getrennt durch ;
  IFS=';' read -ra _extras <<< "$EXTRA_DOWNLOADS"
  for e in "${_extras[@]}"; do
    e="$(echo "$e" | xargs)"; [[ -z "$e" ]] && continue
    IFS='|' read -r e_url e_dir e_name <<< "$e"
    [[ -z "${e_name:-}" ]] && e_name="$(basename "${e_url%%\?*}")"
    auth=""; [[ "$e_url" == *civitai.com* ]] && auth="civitai"; [[ "$e_url" == *huggingface.co* ]] && auth="hf"
    add_file "$e_url" "${e_dir:-loras}" "$e_name" "$auth"
  done
fi

if [[ ! -s "$LIST" ]]; then log "Alles bereits vorhanden - nichts zu laden."; exit 0; fi
start_ts=$(date +%s)
if command -v aria2c >/dev/null 2>&1; then
  log "Starte aria2c (${PARALLEL_DOWNLOADS} Dateien parallel, ${ARIA_CONNECTIONS} Verbindungen je Datei) ..."
  aria2c --input-file="$LIST" \
    --max-concurrent-downloads="$PARALLEL_DOWNLOADS" \
    --max-connection-per-server="$ARIA_CONNECTIONS" --split="$ARIA_CONNECTIONS" \
    --min-split-size=16M --continue=true --auto-file-renaming=false --allow-overwrite=false \
    --file-allocation=none --max-tries=10 --retry-wait=5 --timeout=60 \
    --summary-interval=30 --console-log-level=warn --download-result=full \
    --user-agent="runpod-dasiwa/1.0"
  rc=$?
else
  log "aria2c nicht gefunden - Fallback: curl, nacheinander (langsamer)."
  rc=0; url=""; dir=""; out=""; hdr=""
  flush() {
    [[ -z "$url" ]] && return
    mkdir -p "$dir"
    if [[ -n "$hdr" ]]; then curl -fL --retry 5 -C - -H "$hdr" -o "$dir/$out" "$url" || rc=1
    else curl -fL --retry 5 -C - -o "$dir/$out" "$url" || rc=1; fi
    url=""; dir=""; out=""; hdr=""
  }
  while IFS= read -r line; do
    case "$line" in
      "  dir="*) dir="${line#  dir=}" ;;
      "  out="*) out="${line#  out=}" ;;
      "  header="*) hdr="${line#  header=}" ;;
      *) flush; url="$line" ;;
    esac
  done < "$LIST"
  flush
fi
log "Fertig nach $(( $(date +%s) - start_ts ))s (exit code $rc)."
[[ $rc -ne 0 ]] && log "WARNUNG: Nicht alle Downloads erfolgreich. Erneut: /workspace/dasiwa/bin/download_models.sh"
exit $rc
DL_EOF

cat > "$BIN/start_comfy.sh" <<START_EOF
#!/usr/bin/env bash
# Startet ComfyUI im Hintergrund (Log: $WS/comfyui.log). Laeuft schon eins, passiert nichts.
[[ -f /etc/rp_environment ]] && source /etc/rp_environment
if pgrep -f "main.py --listen 0.0.0.0 --port $COMFY_PORT" >/dev/null; then echo "ComfyUI laeuft bereits."; exit 0; fi
cd "$COMFY_DIR" || exit 1
echo "[dasiwa] \$(date '+%H:%M:%S') Starte ComfyUI auf 0.0.0.0:$COMFY_PORT" >> "$WS/comfyui.log"
# shellcheck disable=SC2086
nohup setsid "$VENV/bin/python" main.py --listen 0.0.0.0 --port $COMFY_PORT --enable-manager \\
  --preview-method auto \${COMFY_ARGS:-} >> "$WS/comfyui.log" 2>&1 < /dev/null &
echo "ComfyUI gestartet (PID \$!), Log: $WS/comfyui.log"
START_EOF

cat > "$BIN/restart_comfy.sh" <<RESTART_EOF
#!/usr/bin/env bash
# Beendet ComfyUI und startet es neu.
pkill -f "main.py --listen 0.0.0.0 --port $COMFY_PORT" && echo "ComfyUI beendet." || echo "ComfyUI lief nicht."
for _ in \$(seq 1 20); do pgrep -f "main.py --listen 0.0.0.0 --port $COMFY_PORT" >/dev/null || break; sleep 1; done
pkill -9 -f "main.py --listen 0.0.0.0 --port $COMFY_PORT" 2>/dev/null || true
exec "$BIN/start_comfy.sh"
RESTART_EOF
chmod +x "$BIN"/*.sh "$BIN"/*.py

# Notebook auf das Volume legen (nur wenn noch nicht vorhanden, damit eigene Aenderungen bleiben)
if [[ ! -f "$WS/setup.ipynb" ]]; then
  if [[ -n "${NOTEBOOK_URL:-}" ]]; then
    curl -fsSL "$NOTEBOOK_URL" -o "$WS/setup.ipynb" && log "Notebook abgelegt: $WS/setup.ipynb" || log "Notebook-Download fehlgeschlagen"
  else
    SELF_DIR_URL="${PROVISION_BASE_URL:-}"
    if [[ -n "$SELF_DIR_URL" ]]; then
      curl -fsSL "$SELF_DIR_URL/setup.ipynb" -o "$WS/setup.ipynb" && log "Notebook abgelegt: $WS/setup.ipynb" || true
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 1) System-Pakete (aria2 fehlt im runpod/pytorch-Image)
# ---------------------------------------------------------------------------
if ! command -v aria2c >/dev/null 2>&1; then
  log "Installiere aria2 ..."
  (apt-get update -qq && apt-get install -y -qq aria2 >/dev/null) || log "WARNUNG: aria2 konnte nicht installiert werden (Fallback curl)."
fi

# ---------------------------------------------------------------------------
# 2) ComfyUI klonen (vor den Downloads, damit models/ existiert)
# ---------------------------------------------------------------------------
if [[ ! -f "$COMFY_DIR/main.py" ]]; then
  log "Klonen: ComfyUI ..."
  git clone -q https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" || { log "FEHLER: ComfyUI-Clone fehlgeschlagen"; exit 0; }
  [[ -n "$COMFY_REF" ]] && git -C "$COMFY_DIR" checkout -q "$COMFY_REF"
elif [[ "$UPDATE_NODES" == "1" ]]; then
  log "Update: ComfyUI (git pull)"; git -C "$COMFY_DIR" pull -q --ff-only || true
fi
for sub in diffusion_models/MiniMaxH3 text_encoders vae/MiniMaxH3 vae_approx loras llm latent_upscale_models upscale_models frame_interpolation; do
  mkdir -p "$MODELS_DIR/$sub"
done
mkdir -p "$COMFY_DIR/user/default/workflows"

# ---------------------------------------------------------------------------
# 3) Downloads im Hintergrund starten (parallel zur Python-Installation)
# ---------------------------------------------------------------------------
rm -f "$MODELS_DIR/.dasiwa_selected_model"
log "Starte Modell-Downloads im Hintergrund - Log: $DASIWA_HOME/download.log"
( "$BIN/download_models.sh" > "$DASIWA_HOME/download.log" 2>&1; echo "[download] beendet (exit $?)" >> "$DASIWA_HOME/download.log" ) &
DL_PID=$!

# ---------------------------------------------------------------------------
# 4) venv auf dem Volume (nutzt Torch aus dem Image via --system-site-packages)
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Erzeuge venv: $VENV"
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel
fi
PY="$VENV/bin/python"; PIP="$VENV/bin/pip"
TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
if [[ "$TORCH_CUDA" != 13* ]]; then
  log "Kein Torch mit CUDA 13 gefunden (gefunden: '${TORCH_CUDA:-keins}') - installiere torch cu130 in den venv ..."
  "$PIP" install -q torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" || log "WARNUNG: Torch-Installation fehlgeschlagen"
fi
log "Torch: $("$PY" -c 'import torch; print(torch.__version__, "CUDA", torch.version.cuda)' 2>&1 | tail -1)"

# pip install nur wenn sich die requirements-Datei geaendert hat (schnelle Neustarts)
pip_req() {  # <requirements-datei> <marker-name> [extra pip args...]
  local req="$1" name="$2"; shift 2
  [[ -f "$req" ]] || return 0
  local sum; sum="$(md5sum "$req" | cut -d' ' -f1)"
  if [[ "$(cat "$DASIWA_HOME/markers/$name" 2>/dev/null)" == "$sum" ]]; then return 0; fi
  log "pip install: $name"
  if "$PIP" install -q -r "$req" "$@"; then echo "$sum" > "$DASIWA_HOME/markers/$name"
  else log "WARNUNG: pip install fuer $name nicht vollstaendig erfolgreich"; fi
}
pip_req "$COMFY_DIR/requirements.txt" comfyui
pip_req "$COMFY_DIR/manager_requirements.txt" comfyui-manager

# ---------------------------------------------------------------------------
# 5) Custom Nodes
# ---------------------------------------------------------------------------
clone_node() {  # <repo-url>
  local url="$1" dir; dir="$COMFY_DIR/custom_nodes/$(basename "$url" .git)"
  if [[ ! -d "$dir/.git" ]]; then
    log "Klonen: $(basename "$dir")"; git clone -q "$url" "$dir" || log "FEHLER: Clone $url"
  elif [[ "$UPDATE_NODES" == "1" ]]; then
    git -C "$dir" pull -q --ff-only || true
  fi
}
clone_node https://github.com/kijai/ComfyUI-KJNodes.git
clone_node https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git
pip_req "$COMFY_DIR/custom_nodes/ComfyUI-KJNodes/requirements.txt" kjnodes

DREQ="$COMFY_DIR/custom_nodes/ComfyUI-DaSiWa-Nodes/requirements.txt"
if [[ -f "$DREQ" ]]; then
  # llama-cpp-python fuer "Prompt Forge" mit GGUF-Modell: fertiges CUDA-13-Wheel (~780 MB, einmalig
  # in den venv). Ist schon eine Version MIT GPU-Offload installiert, passiert nichts.
  FORGE_MODEL="${FORGE_MODEL:-gguf-8b-q8}"
  if [[ "$FORGE_MODEL" != none && "$FORGE_MODEL" != 0 && "$FORGE_MODEL" != false && "$FORGE_MODEL" != hf-* ]]; then
    if [[ "$("$PY" -c 'import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())' 2>/dev/null)" != "True" ]]; then
      log "Installiere llama-cpp-python (CUDA 13, ${LLAMA_CPP_INDEX:-cu130}) ..."
      if "$PIP" install -q --force-reinstall --no-deps --only-binary=llama-cpp-python \
           --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${LLAMA_CPP_INDEX:-cu130}" \
           "llama-cpp-python>=0.3.26"; then
        "$PIP" install -q "typing-extensions>=4.5.0" "numpy>=1.20.0" "diskcache>=5.6.1" "jinja2>=2.11.3" || true
      else
        log "WARNUNG: llama-cpp-python (CUDA) nicht installiert - GGUF im Prompt Forge nicht nutzbar (FORGE_MODEL=hf-8b als Alternative)."
      fi
      log "llama-cpp: $("$PY" -c 'import llama_cpp; print(llama_cpp.__version__, "GPU-Offload:", llama_cpp.llama_supports_gpu_offload())' 2>&1 | tail -1)"
    fi
  fi
  grep -v -E '^(llama-cpp-python|amd-adlx)' "$DREQ" > "$DASIWA_HOME/dasiwa-req.txt"
  pip_req "$DASIWA_HOME/dasiwa-req.txt" dasiwa-nodes --extra-index-url https://pypi.nvidia.com
fi
for d in "$COMFY_DIR"/custom_nodes/*/; do
  n="$(basename "$d")"; case "$n" in ComfyUI-KJNodes|ComfyUI-DaSiWa-Nodes|__pycache__) continue ;; esac
  pip_req "$d/requirements.txt" "node-$n"    # per Manager nachinstallierte Nodes
done

# ---------------------------------------------------------------------------
# 6) Workflow ablegen (Dateinamen an gewaehlte Variante anpassen)
# ---------------------------------------------------------------------------
WF_CACHE="$DASIWA_HOME/workflows"; mkdir -p "$WF_CACHE"
wf_name="$WF_DEFAULT_FILE"
if [[ "$UPDATE_WORKFLOW" == "1" ]]; then
  latest="$(curl -fsSL "https://api.github.com/repos/darksidewalker/dasiwa-comfyui-workflows/contents/C-MMH3" \
    | jq -r '[.[] | select(.name|endswith(".json"))] | sort_by(.name | capture("-(?<n>[0-9]+)\\.json$").n | tonumber) | last | .name' 2>/dev/null)"
  [[ -n "$latest" && "$latest" != "null" ]] && wf_name="$latest"
fi
if [[ ! -s "$WF_CACHE/$wf_name" ]]; then
  log "Lade Workflow: $wf_name"
  curl -fsSL "$WF_REPO_RAW/${wf_name// /%20}" -o "$WF_CACHE/$wf_name" || { log "FEHLER: Workflow-Download"; rm -f "$WF_CACHE/$wf_name"; }
fi
for _ in $(seq 1 120); do [[ -f "$MODELS_DIR/.dasiwa_selected_model" ]] && break; sleep 1; done
if [[ -s "$WF_CACHE/$wf_name" ]]; then
  SEL_MODEL="$(cat "$MODELS_DIR/.dasiwa_selected_model" 2>/dev/null)" \
  SEL_TE="$(cat "$MODELS_DIR/.dasiwa_selected_te" 2>/dev/null)" \
  SEL_VAE="$(cat "$MODELS_DIR/.dasiwa_selected_vae" 2>/dev/null)" \
  python3 - "$WF_CACHE/$wf_name" "$COMFY_DIR/user/default/workflows/$wf_name" <<'PY'
import os, sys
s = open(sys.argv[1], encoding="utf-8").read()
m, te, vae = os.environ.get("SEL_MODEL"), os.environ.get("SEL_TE"), os.environ.get("SEL_VAE")
if m: s = s.replace("MiniMaxH3/dasiwa_minimax_h3_ref2va_v4_pruned_turbo_int8.safetensors", m)
if te: s = s.replace("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", te)
if vae == "fp16": s = s.replace("MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors", "MiniMaxH3/minimax_h3_video_vae_fp16.safetensors")
open(sys.argv[2], "w", encoding="utf-8").write(s)
PY
  log "Workflow bereit: user/default/workflows/$wf_name"
fi

# ---------------------------------------------------------------------------
# 7) ComfyUI starten
# ---------------------------------------------------------------------------
if [[ "$START_EARLY" != "1" ]]; then
  log "Warte auf Downloads (START_EARLY=0) ..."
  wait "$DL_PID"
fi
"$BIN/start_comfy.sh"
log "ComfyUI: Port $COMFY_PORT (Log: $WS/comfyui.log)"

if [[ "$START_EARLY" == "1" ]]; then
  wait "$DL_PID"
  log "Downloads abgeschlossen - im ComfyUI-Browser 'R' druecken, um Modell-Listen neu zu laden."
fi
tail -n 3 "$DASIWA_HOME/download.log" | sed 's/^/[dasiwa]   /'
log "===================== Provisioning fertig ====================="
exit 0
