"""Fase D interactive demo (Etapa 1): chat with Cracked-D, the decoder-only model
of the controlled decoder-only vs encoder-decoder comparison. Same Streamlit app
as the Fase A/B/C demos -- Streamlit auto-discovers pages/*.py, no extra
deployment. Sliced-D is added here in Etapa 2 (with a model switcher, like Fase C).

All load/generation logic lives in compare_lab.demo (Streamlit-free, unit-tested);
this file is only UI. The model is loaded lazily via st.cache_resource, so it is
only downloaded/held in memory when this page is actually opened.

Names in the UI are Cracked-D / Sliced-D (not the Fase C Cracked/Sliced), because
they are different models: retrained from scratch on SmolLM-Corpus + smol-smoltalk
with a matched parameter budget. It is a ~26M-parameter model -- answers will be
limited.
"""

import base64
import html
import time
from pathlib import Path

import streamlit as st

from compare_lab.demo import load_model, chat
from coconut_lab.logos import CRACKED_WORDMARK

ASSETS_DIR = Path(__file__).resolve().parent.parent / "coconut_tui" / "assets"
LOGO_FONT_PATH = ASSETS_DIR / "DejaVuSansMono.ttf"
OUTPUT_BOX_HEIGHT = 480

st.set_page_config(
    page_title="Fase D - Comparacion Controlada", page_icon="⚖️", layout="wide",
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
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 900px; margin: 0 auto; }}

.mobile-only {{ display: block !important; }}
.desktop-only {{ display: none !important; }}
@media (min-width: 768px) {{
    .mobile-only {{ display: none !important; }}
    .desktop-only {{ display: block !important; }}
}}
.model-banner {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin: 0 0 0.75rem 0; }}
.model-banner-gif {{ width: 130px; height: auto; flex: 0 0 auto; }}
@media (min-width: 768px) {{ .model-banner-gif {{ width: 180px; }} }}
.model-wordmark-text {{
    font-size: 0.5rem; line-height: 1.15; white-space: pre; overflow-x: auto;
    color: #c98a4b; margin: 0; font-family: "CoconutLogoFont", monospace;
}}
@media (min-width: 1000px) {{ .model-wordmark-text {{ font-size: 0.62rem; }} }}
.model-wordmark-img {{ max-width: 200px; width: 55%; height: auto; }}
.coconut-caption {{ color: #7b8790; font-size: 0.85rem; margin-bottom: 0.75rem; }}
.coconut-info {{
    color: #7b8790; border-left: 2px solid #8a6238; padding: 0.4rem 0.75rem;
    margin-bottom: 1rem; font-size: 0.85rem; line-height: 1.5;
}}
.coconut-info strong {{ color: #c98a4b; }}

.st-key-output-panel {{
    background: #12171a !important; border: 1px solid #23292c !important; border-radius: 10px !important;
}}
@media (max-width: 767px) {{
    .st-key-output-panel {{ height: 360px !important; max-height: 360px !important; }}
    div[data-testid="stLayoutWrapper"]:has(> .st-key-output-panel) {{
        flex: 0 0 360px !important; height: 360px !important;
    }}
}}

.chat-row {{ display: flex; margin: 0.45rem 0; }}
.chat-row.user {{ justify-content: flex-end; }}
.chat-row.assistant {{ justify-content: flex-start; }}
.chat-bubble {{
    max-width: 75%; padding: 0.55rem 0.9rem; border-radius: 16px;
    line-height: 1.5; font-size: 0.92rem; white-space: pre-wrap; word-break: break-word;
    font-family: inherit;
}}
.chat-bubble.user {{ background: #1f6f4a; color: #eafff2; border-bottom-right-radius: 4px; }}
.chat-bubble.assistant {{
    background: #1a2124; color: #d8dee2; border: 1px solid #23292c; border-bottom-left-radius: 4px;
}}

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
.st-key-prompt-row {{ margin-top: -0.75rem; }}
.st-key-prompt-row div[data-testid="stHorizontalBlock"] {{
    flex-direction: row !important; flex-wrap: nowrap !important;
    align-items: center !important; gap: 0.5rem !important;
}}
.st-key-prompt-row div[data-testid="stColumn"] {{ min-width: 0 !important; }}
.st-key-prompt-row div.stButton > button {{
    background: #1f6f4a; color: #eafff2; border: 1px solid #2a8a5e;
    border-radius: 999px; white-space: nowrap; padding: 0.35rem 1rem;
}}
.st-key-prompt-row div.stButton > button:hover {{
    background: #2a8a5e; border-color: #6fcf97; color: #ffffff;
}}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando Cracked-D (solo la primera vez)...")
def get_model():
    # local checkpoint if present, otherwise the Hub (see compare_lab.demo)
    return load_model()


def _bubble(role: str, text: str) -> str:
    body = html.escape(text).replace("\n", "<br>")
    return f'<div class="chat-row {role}"><div class="chat-bubble {role}">{body}</div></div>'


def render_conversation(history: list[dict], cursor_text: str | None = None) -> str:
    parts = [_bubble(t["role"], t["content"]) for t in history]
    if cursor_text is not None:
        parts.append(_bubble("assistant", cursor_text + "▌"))
    return "".join(parts)


# --- banner (reuses Cracked's GIF + wordmark from Fase C assets) ---
_gif_b64 = base64.b64encode((ASSETS_DIR / "cracked.gif").read_bytes()).decode("ascii")
_wordmark_b64 = base64.b64encode((ASSETS_DIR / "cracked_wordmark.png").read_bytes()).decode("ascii")
_wordmark_lines_html = "<br>".join(html.escape(line) for line in CRACKED_WORDMARK.splitlines())
st.markdown(
    '<div class="model-banner">'
    f'<img class="model-banner-gif" src="data:image/gif;base64,{_gif_b64}" alt="Cracked-D">'
    f'<div class="model-wordmark-text desktop-only">{_wordmark_lines_html}</div>'
    f'<img class="model-wordmark-img mobile-only" src="data:image/png;base64,{_wordmark_b64}" alt="Cracked-D">'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="coconut-caption">Cracked-D &middot; decoder-only &middot; comparacion controlada de Fase D</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="coconut-info">Fase D reentrena desde cero, con los mismos datos, tokenizer y '
    'presupuesto, un <strong>decoder-only (Cracked-D)</strong> y un encoder-decoder (Sliced-D, en '
    'Etapa 2) de ~26M parametros, preentrenados en SmolLM-Corpus y afinados en smol-smoltalk. '
    'Es un modelo <strong>muy pequeno (~26M)</strong>: sus respuestas seran limitadas. '
    'Detalles y resultados en el README.</div>',
    unsafe_allow_html=True,
)

if st.button("Reiniciar conversacion", use_container_width=True):
    st.session_state.fased_history = []
    st.session_state.fased_last_processed = None
    st.rerun()

for key, default in (("fased_history", []), ("fased_last_processed", None),
                     ("fased_clear_input", False)):
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.fased_clear_input:
    st.session_state.fased_prompt = ""
    st.session_state.fased_clear_input = False

model, tok, device = get_model()

st.markdown(
    '<div class="terminal-head-bar" style="display:flex;align-items:center;gap:.35rem;'
    'background:#12171a;border:1px solid #23292c;border-bottom:none;border-radius:10px 10px 0 0;'
    'padding:.5rem .8rem;color:#7b8790;font-size:.78rem;margin-top:.5rem;">'
    f"&#9679;&#9679;&#9679;&nbsp;Cracked-D &mdash; streamlit &middot; {device}</div>",
    unsafe_allow_html=True,
)
with st.container(height=OUTPUT_BOX_HEIGHT, border=True, key="output-panel"):
    output_box = st.empty()
    output_box.markdown(render_conversation(st.session_state.fased_history), unsafe_allow_html=True)

with st.container(key="prompt-row"):
    caret_col, input_col, send_col = st.columns([0.05, 0.78, 0.17])
    with caret_col:
        st.markdown('<div class="prompt-caret">&gt;</div>', unsafe_allow_html=True)
    with input_col:
        prompt = st.text_input("Mensaje", key="fased_prompt", placeholder="escribe algo...",
                               label_visibility="collapsed")
    with send_col:
        send_clicked = st.button("Enviar", use_container_width=True)

if prompt.strip() and (send_clicked or prompt != st.session_state.fased_last_processed):
    st.session_state.fased_history.append({"role": "user", "content": prompt})
    output_box.markdown(render_conversation(st.session_state.fased_history, cursor_text=""),
                        unsafe_allow_html=True)

    full_response = chat(model, tok, st.session_state.fased_history, device=device, max_new_tokens=200)

    words = full_response.split(" ")
    for i in range(len(words)):
        revealed = " ".join(words[:i + 1])
        output_box.markdown(render_conversation(st.session_state.fased_history, cursor_text=revealed),
                            unsafe_allow_html=True)
        time.sleep(0.03)

    st.session_state.fased_history.append({"role": "assistant", "content": full_response})
    st.session_state.fased_last_processed = prompt
    st.session_state.fased_clear_input = True
    output_box.markdown(render_conversation(st.session_state.fased_history), unsafe_allow_html=True)
    st.rerun()

with st.sidebar:
    st.header("Fase D")
    st.markdown(
        "Comparacion controlada decoder-only (**Cracked-D**) vs encoder-decoder "
        "(**Sliced-D**, Etapa 2), con mismos datos, tokenizer, parametros (~26M) y "
        "presupuesto de entrenamiento. Preentrenado en SmolLM-Corpus, afinado en smol-smoltalk."
    )
    st.markdown("---")
    st.markdown("[Codigo en GitHub](https://github.com/davidmorgadocarames/mini_llms)")
    st.caption("build: fase-d.1")
