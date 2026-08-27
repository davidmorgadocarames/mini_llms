"""Streamlit Community Cloud app: real, live inference with the trained
Fase A checkpoint (downloaded from the HuggingFace Hub model repo
davidmorgado/coconut-mini-llm at startup, then cached).

This is the "real inference in the browser" counterpart to the static replay
demo at docs/index.html -- here the model actually runs, on Streamlit's
server, not a pre-recorded transcript. Styled like a terminal (same visual
language as coconut_tui and docs/index.html) instead of a plain web form,
with a live token counter -- both requested explicitly after the first
version looked too much like a generic Streamlit form.
"""

import base64
import html
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import streamlit as st
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"
LOGO_PATH = Path(__file__).resolve().parent / "coconut_tui" / "assets" / "logo.png"

EXAMPLE_PROMPTS = [
    "The history of the",
    "In 1943, the",
    "The film received",
    "The album was praised by critics for its",
]

st.set_page_config(page_title="Coconut mini-LLM", page_icon="🥥")

TERMINAL_CSS = """
<style>
.stApp { background-color: #0b0e0f; }
html, body, [class*="css"], .stApp, .stMarkdown, .stButton button,
.stTextArea textarea, .stSlider label, .stSlider [data-testid="stTickBar"] {
    font-family: ui-monospace, SFMono-Regular, "Cascadia Code", "Fira Code",
                 Consolas, "Courier New", monospace !important;
}
.stApp, .stApp p, .stApp label, .stApp span { color: #d8dee2; }
[data-testid="stSidebar"] { background-color: #12171a; border-right: 1px solid #23292c; }

/* Pre-rendered PNG instead of live block-character text: mobile browsers
   substitute generic "tofu" glyphs for Unicode block-drawing characters
   (confirmed on a real device), so this renders identically everywhere. */
.coconut-banner-img {
    max-width: 420px;
    width: 100%;
    height: auto;
    display: block;
    margin: 0 0 0.75rem 0;
}
.coconut-caption { color: #7b8790; font-size: 0.85rem; margin-bottom: 0.75rem; }
.coconut-info {
    color: #7b8790; border-left: 2px solid #8a6238; padding: 0.4rem 0.75rem;
    margin-bottom: 1rem; font-size: 0.85rem; line-height: 1.5;
}
.coconut-info strong { color: #c98a4b; }

div.stButton > button {
    background: #1a2124; color: #d8dee2; border: 1px solid #23292c;
    border-radius: 6px; font-family: inherit;
}
div.stButton > button:hover { border-color: #8a6238; color: #c98a4b; }
div.stButton > button[kind="primary"] { background: #1f3d2b; border-color: #6fcf97; color: #6fcf97; }

.stTextArea textarea {
    background: #12171a !important; color: #d8dee2 !important;
    border: 1px solid #23292c !important; font-family: inherit !important;
}

.token-counter {
    color: #6fcf97; font-size: 0.82rem; margin: 0.4rem 0 0.75rem 0;
}
pre.coconut-output {
    background: #12171a; border: 1px solid #23292c; border-radius: 8px;
    padding: 0.9rem; white-space: pre-wrap; word-break: break-word;
    line-height: 1.55; font-size: 0.92rem; color: #d8dee2;
}
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Descargando y cargando el modelo (puede tardar un minuto la primera vez)...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = hf_hub_download(HF_REPO, "ckpt.pt")
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")

    tokenizer = BPETokenizer(vocab_path, merges_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, device, checkpoint.get("step")


model, tokenizer, device, step = load_model()

_logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
st.markdown(
    f'<img class="coconut-banner-img" src="data:image/png;base64,{_logo_b64}" alt="Coconut">',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="coconut-caption">Coconut &middot; {model.num_parameters() / 1e6:.1f}M params '
    f"&middot; step {step:,} &middot; corriendo en <code>{device}</code></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="coconut-info"><strong>Coconut es un modelo base</strong> (sin fine-tuning de '
    "instrucciones): escribe el principio de una frase para que la continue, no le hagas "
    "preguntas directas -- funciona mejor con prosa tipo Wikipedia que con conversacion.</div>",
    unsafe_allow_html=True,
)

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

token_counter = st.empty()


def render_token_counter(current: int = 0) -> None:
    session_total = st.session_state.total_tokens + current
    token_counter.markdown(
        f'<div class="token-counter">&#9679; tokens generados: {current} '
        f"(sesion: {session_total})</div>",
        unsafe_allow_html=True,
    )


render_token_counter()

with st.sidebar:
    st.header("Parametros")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("Top-k", 1, 200, 50, 1)
    max_new_tokens = st.slider("Tokens a generar", 20, 400, 150, 10)
    st.markdown("---")
    st.markdown("[Codigo en GitHub](https://github.com/davidmorgadocarames/mini_llms)")

st.markdown("**Prueba, por ejemplo:**")
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

cols = st.columns(len(EXAMPLE_PROMPTS))
for col, example in zip(cols, EXAMPLE_PROMPTS):
    if col.button(example, use_container_width=True):
        st.session_state.prompt = example

prompt = st.text_area(
    "Prompt", key="prompt", height=80, placeholder="Escribe el principio de una frase..."
)

output_box = st.empty()

if st.button("Generar", type="primary") and prompt.strip():
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    text_so_far = tokenizer.decode(ids)
    n_tokens = 0

    output_box.markdown(
        f'<pre class="coconut-output">{html.escape(text_so_far)}&#9608;</pre>',
        unsafe_allow_html=True,
    )

    for out_idx in model.generate_stream(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    ):
        text_so_far = tokenizer.decode(out_idx[0].tolist())
        n_tokens += 1
        output_box.markdown(
            f'<pre class="coconut-output">{html.escape(text_so_far)}&#9608;</pre>',
            unsafe_allow_html=True,
        )
        render_token_counter(n_tokens)

    output_box.markdown(f'<pre class="coconut-output">{html.escape(text_so_far)}</pre>', unsafe_allow_html=True)
    st.session_state.total_tokens += n_tokens
    render_token_counter()
