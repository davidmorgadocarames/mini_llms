"""Fase C interactive demo: chat with any of the three architectures
(Cracked/Sliced/Pressed) trained and compared in C.1-C.6, and switch
between them mid-conversation. Same Streamlit app as the Fase A/B demos
(streamlit_app.py, pages/1_Fase_B_Depth_Lab.py) -- just a third page,
Streamlit auto-discovers pages/*.py, no extra deployment needed.

Model weights are the "final" checkpoints coconut_lab/eval/run_eval.py
trained and evaluated in C.6, hosted on the same HuggingFace Hub model
repo as Fase A/B, under a coconut_lab/ prefix.

Generation: Cracked and Sliced both support real token-by-token streaming
(GPT.generate_stream / EncoderDecoderTransformer.generate_stream), but
Pressed structurally can't -- its replacer only corrects a draft that's
already been generated in full, so there's nothing to show until the whole
pipeline finishes. To keep the feel consistent across all three, every
response is computed in full first, then revealed word by word with a
short delay -- the same "compute eagerly, replay with time.sleep" trick
already used for Fase B's animated LLR trace (pages/1_Fase_B_Depth_Lab.py).
"""

import base64
import html
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import streamlit as st
from coconut_lab.data.prepare_conversations import ASSISTANT_MARKER, format_turns
from coconut_lab.logos import CRACKED_LOGO, PRESSED_LOGO, SLICED_LOGO
from coconut_lab.models import cracked as cracked_mod
from coconut_lab.models import sliced as sliced_mod
from coconut_lab.models.pressed_loop import reduce_with_pressed
from depth_lab.models.encoder_decoder import EncDecConfig, EncoderDecoderTransformer
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.replacer import Replacer, ReplacerConfig
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "coconut_tui" / "assets"
LOGO_FONT_PATH = ASSETS_DIR / "DejaVuSansMono.ttf"
OUTPUT_BOX_HEIGHT = 480
DRAFTER_BLOCK_SIZE = 512  # matches coconut_lab.models.pressed.DRAFTER_BLOCK_SIZE

MODELS = {
    "cracked": {
        "label": "Cracked",
        "logo": CRACKED_LOGO,
        "png": "cracked_logo.png",
        "desc": "Decoder-only (Coconut afinado sobre instrucciones). El baseline.",
    },
    "sliced": {
        "label": "Sliced",
        "logo": SLICED_LOGO,
        "png": "sliced_logo.png",
        "desc": "Encoder-decoder entrenado desde cero, sin preentrenamiento previo.",
    },
    "pressed": {
        "label": "Pressed",
        "logo": PRESSED_LOGO,
        "png": "pressed_logo.png",
        "desc": "Looped Locate-and-Replace: redacta un borrador y corrige su aritmetica.",
    },
}
DEFAULT_MODEL = "cracked"  # see README/CLAUDE.md Fase C: gana el punto mas importante de C.6
                            # (set propio de dominio) y lambada_openai; margen pequeno frente
                            # a Pressed en piqa/k-fold -- no es un resultado aplastante.

st.set_page_config(
    page_title="Fase C - Coconut Interactivo", page_icon="\U0001f3ae", layout="wide",
    initial_sidebar_state="collapsed",
)

_logo_font_b64 = base64.b64encode(LOGO_FONT_PATH.read_bytes()).decode("ascii")
PAGE_CSS = f"""
<style>
@font-face {{
    font-family: "CoconutLogoFont";
    src: url(data:font/ttf;base64,{_logo_font_b64}) format("truetype");
    font-display: block;
}}
.stApp {{ background-color: #0b0e0f; }}
html, body, [class*="css"], .stApp, .stMarkdown, .stButton button, .stTextInput input {{
    font-family: ui-monospace, SFMono-Regular, "Cascadia Code", "Fira Code",
                 Consolas, "Courier New", monospace !important;
}}
.stApp, .stApp p, .stApp label, .stApp span {{ color: #d8dee2; }}
[data-testid="stSidebar"] {{ background-color: #12171a; border-right: 1px solid #23292c; }}
[data-testid="stHeader"] {{ background: #0b0e0f; }}
[data-testid="stToolbarActions"] {{ display: none !important; }}
[data-testid="stMainMenuButton"] {{ display: none !important; }}
[data-testid="stAppDeployButton"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stExpandSidebarButton"] {{ color: #c98a4b !important; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 900px; margin: 0 auto; }}

.mobile-only {{ display: block !important; }}
.desktop-only {{ display: none !important; }}
@media (min-width: 768px) {{
    .mobile-only {{ display: none !important; }}
    .desktop-only {{ display: block !important; }}
}}
.coconut-banner-img {{ max-width: 320px; width: 100%; height: auto; margin: 0 0 0.75rem 0; }}
.coconut-logo-text {{
    font-size: 0.55rem; line-height: 1.15; white-space: pre; overflow-x: auto;
    color: #c98a4b; margin: 0 0 0.75rem 0; font-family: "CoconutLogoFont", monospace;
}}
@media (min-width: 1000px) {{ .coconut-logo-text {{ font-size: 0.68rem; }} }}
.coconut-caption {{ color: #7b8790; font-size: 0.85rem; margin-bottom: 0.75rem; }}
.coconut-info {{
    color: #7b8790; border-left: 2px solid #8a6238; padding: 0.4rem 0.75rem;
    margin-bottom: 1rem; font-size: 0.85rem; line-height: 1.5;
}}
.coconut-info strong {{ color: #c98a4b; }}

.st-key-output-panel {{
    background: #12171a !important; border: 1px solid #23292c !important; border-radius: 10px !important;
}}
@media (max-width: 767px) {{ .st-key-output-panel {{ max-height: 380px !important; }} }}

#output-frame {{
    white-space: pre-wrap !important; word-break: break-word;
    line-height: 1.55; font-size: 0.92rem; color: #d8dee2; margin: 0; font-family: inherit;
}}
#output-frame .prompt-echo {{ color: #6fcf97; }}

.prompt-caret {{ color: #6fcf97; font-weight: 700; font-size: 1.1rem; display: flex;
    align-items: center; height: 2.6rem; }}
.stTextInput input {{
    background: transparent !important; color: #d8dee2 !important;
    border: none !important; border-bottom: 1px solid #23292c !important;
    border-radius: 0 !important; font-family: inherit !important;
}}
div.stButton > button {{
    background: #1a2124; color: #d8dee2; border: 1px solid #23292c;
    border-radius: 8px; font-family: inherit;
}}
div.stButton > button:hover {{ border-color: #8a6238; color: #c98a4b; }}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Descargando los modelos de Fase C (solo la primera vez)...")
def load_models():
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    tokenizer = BPETokenizer(vocab_path, merges_path)

    def _load(filename, config_cls, model_cls):
        path = hf_hub_download(HF_REPO, f"coconut_lab/{filename}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = model_cls(config_cls(**ckpt["config"])).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    cracked = _load("cracked_final.pt", GPTConfig, GPT)
    sliced = _load("sliced_final.pt", EncDecConfig, EncoderDecoderTransformer)
    drafter = _load("pressed_drafter_final.pt", GPTConfig, GPT)
    locator = _load("pressed_locator.pt", LocatorConfig, Locator)
    replacer = _load("pressed_replacer.pt", ReplacerConfig, Replacer)
    return tokenizer, cracked, sliced, drafter, locator, replacer


tokenizer, cracked, sliced, drafter, locator, replacer = load_models()


def build_prompt(history: list[dict], block_size: int, reserve_tokens: int = 200) -> str:
    """Renders history + a trailing "<|assistant|>\\n" cue, dropping the
    oldest turns first if it would overflow block_size -- same "truncate
    from the start, keep the most recent turns" rule
    coconut_lab.data.loader.ConversationDataset trains under, reapplied
    here so a long chat degrades gracefully instead of crashing."""
    max_prompt_tokens = block_size - reserve_tokens
    turns = history[:]
    while turns:
        text = format_turns(turns) + ASSISTANT_MARKER
        if len(tokenizer.encode(text)) <= max_prompt_tokens:
            return text
        turns = turns[1:]
    return ASSISTANT_MARKER


def generate_full_response(model_id: str, prompt_text: str) -> str:
    if model_id == "cracked":
        return cracked_mod.generate_response(cracked, tokenizer, prompt_text, DEVICE, max_new_tokens=150)
    if model_id == "sliced":
        return sliced_mod.generate_response(sliced, tokenizer, prompt_text, DEVICE, max_new_tokens=150)
    return reduce_with_pressed(drafter, locator, replacer, tokenizer, prompt_text, DEVICE,
                                max_new_tokens_draft=200).final_text


@st.dialog("Elegir modelo")
def model_picker():
    for model_id, info in MODELS.items():
        current = " (actual)" if model_id == st.session_state.fasec_model else ""
        if st.button(f"{info['label']}{current} — {info['desc']}", key=f"pick-{model_id}",
                     use_container_width=True):
            st.session_state.fasec_model = model_id
            st.rerun()


for key, default in (
    ("fasec_model", DEFAULT_MODEL),
    ("fasec_history", []),
    ("fasec_last_processed", None),
    ("fasec_clear_input", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Streamlit forbids reassigning a widget-bound session_state key in the same
# run the widget was created in -- so the input box is cleared here, before
# st.text_input(key="fasec_prompt") below runs, on the rerun right after a
# message was sent (flag set at the end of the processing block).
if st.session_state.fasec_clear_input:
    st.session_state.fasec_prompt = ""
    st.session_state.fasec_clear_input = False

active = MODELS[st.session_state.fasec_model]

_logo_b64 = base64.b64encode((ASSETS_DIR / active["png"]).read_bytes()).decode("ascii")
st.markdown(
    f'<img class="coconut-banner-img mobile-only" src="data:image/png;base64,{_logo_b64}" alt="{active["label"]}">',
    unsafe_allow_html=True,
)
_logo_lines_html = "<br>".join(html.escape(line) for line in active["logo"].splitlines())
st.markdown(f'<div class="coconut-logo-text desktop-only">{_logo_lines_html}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="coconut-caption">{active["label"]} &middot; {active["desc"]} &middot; '
    f"corriendo en <code>{DEVICE}</code></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="coconut-info">Fase C compara 3 arquitecturas afinadas para responder '
    "instrucciones y chatear (no solo continuar prosa, como el Coconut base de Fase A). "
    "<strong>Cracked</strong> destaca por defecto porque gana el punto mas importante de la "
    "comparacion (set propio de dominio) -- pero el margen es pequeno en varios puntos, "
    "no un resultado aplastante. Detalles en el README.</div>",
    unsafe_allow_html=True,
)

col_pick, col_reset = st.columns([0.5, 0.5])
with col_pick:
    if st.button("Cambiar modelo", use_container_width=True):
        model_picker()
with col_reset:
    if st.button("Reiniciar conversacion", use_container_width=True):
        st.session_state.fasec_history = []
        st.session_state.fasec_last_processed = None
        st.rerun()


def render_conversation(history: list[dict], cursor_text: str | None = None) -> str:
    if not history and cursor_text is None:
        return '<pre id="output-frame"></pre>'
    parts = []
    for turn in history:
        if turn["role"] == "user":
            parts.append(f'<span class="prompt-echo">&gt; {html.escape(turn["text"])}</span>')
        else:
            parts.append(html.escape(turn["text"]).replace("\n\n", "<br><br>"))
    if cursor_text is not None:
        parts.append(html.escape(cursor_text) + "&#9608;")
    return '<pre id="output-frame">' + "\n\n".join(parts) + "</pre>"


st.markdown(
    '<div class="terminal-head-bar" style="display:flex;align-items:center;gap:.35rem;'
    'background:#12171a;border:1px solid #23292c;border-bottom:none;border-radius:10px 10px 0 0;'
    'padding:.5rem .8rem;color:#7b8790;font-size:.78rem;margin-top:.5rem;">'
    f"&#9679;&#9679;&#9679;&nbsp;{active['label']} &mdash; streamlit</div>",
    unsafe_allow_html=True,
)
with st.container(height=OUTPUT_BOX_HEIGHT, border=True, key="output-panel"):
    output_box = st.empty()
    output_box.markdown(render_conversation(st.session_state.fasec_history), unsafe_allow_html=True)

caret_col, input_col = st.columns([0.03, 0.97])
with caret_col:
    st.markdown('<div class="prompt-caret">&gt;</div>', unsafe_allow_html=True)
with input_col:
    prompt = st.text_input(
        "Mensaje", key="fasec_prompt", placeholder="escribe algo y pulsa Enter",
        label_visibility="collapsed",
    )

if prompt.strip() and prompt != st.session_state.fasec_last_processed:
    st.session_state.fasec_history.append({"role": "user", "text": prompt})
    output_box.markdown(render_conversation(st.session_state.fasec_history, cursor_text=""),
                         unsafe_allow_html=True)

    model_id = st.session_state.fasec_model
    block_size = cracked.config.block_size if model_id == "cracked" else (
        sliced_mod.SRC_BLOCK_SIZE if model_id == "sliced" else DRAFTER_BLOCK_SIZE)
    prompt_text = build_prompt(st.session_state.fasec_history, block_size)
    full_response = generate_full_response(model_id, prompt_text)

    words = full_response.split(" ")
    for i in range(len(words)):
        revealed = " ".join(words[:i + 1])
        output_box.markdown(render_conversation(st.session_state.fasec_history, cursor_text=revealed),
                             unsafe_allow_html=True)
        time.sleep(0.03)

    st.session_state.fasec_history.append({"role": "assistant", "text": full_response})
    st.session_state.fasec_last_processed = prompt
    st.session_state.fasec_clear_input = True
    output_box.markdown(render_conversation(st.session_state.fasec_history), unsafe_allow_html=True)
    st.rerun()

with st.sidebar:
    st.header("Fase C")
    st.markdown(
        "Chat con 3 arquitecturas afinadas sobre instrucciones (Alpaca + oasst1) y "
        "razonamiento (GSM8K): **Cracked** (decoder-only), **Sliced** (encoder-decoder), "
        "**Pressed** (Looped Locate-and-Replace)."
    )
    st.markdown("---")
    st.markdown("[Codigo en GitHub](https://github.com/davidmorgadocarames/mini_llms)")
    st.caption("build: fase-c.1")
