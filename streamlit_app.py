"""Streamlit Community Cloud app: real, live inference with the trained
Fase A checkpoint (downloaded from the HuggingFace Hub model repo
davidmorgado/coconut-mini-llm at startup, then cached).

This is the "real inference in the browser" counterpart to the static replay
demo at docs/index.html -- same visual language and layout (fixed-height
scrolling output panel, token counter + prompt always visible below it,
green prompt echo, pill-shaped wrapping chips, Enter-to-submit with no
separate button), but the model actually runs here instead of replaying a
pre-recorded transcript.
"""

import base64
import html
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import streamlit as st
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer, clean_for_display

HF_REPO = "davidmorgado/coconut-mini-llm"
LOGO_PATH = Path(__file__).resolve().parent / "coconut_tui" / "assets" / "logo.png"

EXAMPLE_PROMPTS = [
    "Once upon a time",
    "The history of the",
    "In 1943, the",
    "The film received",
    "The album was praised by critics for its",
    "The war began when",
]

st.set_page_config(page_title="Coconut mini-LLM", page_icon="🥥")

TERMINAL_CSS = """
<style>
.stApp { background-color: #0b0e0f; }
html, body, [class*="css"], .stApp, .stMarkdown, .stButton button,
.stTextInput input, .stSlider label, .stSlider [data-testid="stTickBar"] {
    font-family: ui-monospace, SFMono-Regular, "Cascadia Code", "Fira Code",
                 Consolas, "Courier New", monospace !important;
}
.stApp, .stApp p, .stApp label, .stApp span { color: #d8dee2; }
[data-testid="stSidebar"] { background-color: #12171a; border-right: 1px solid #23292c; }

/* Hide Streamlit's own chrome (visible to every visitor, not just the app
   owner): the light-colored top header/toolbar and the "Made with
   Streamlit" footer badge. */
[data-testid="stHeader"] { background: #0b0e0f; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; }
.stApp > header { background: transparent; }
.block-container { padding-top: 2rem; }

.coconut-banner-img {
    max-width: 320px;
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

.terminal-head-bar {
    display: flex; align-items: center; gap: .35rem;
    background: #12171a; border: 1px solid #23292c; border-bottom: none;
    border-radius: 10px 10px 0 0; padding: .5rem .8rem;
    color: #7b8790; font-size: .78rem; margin-top: .5rem;
}
.terminal-head-bar .dot {
    width: .55rem; height: .55rem; border-radius: 50%; background: #3a4145;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #12171a !important;
    border: 1px solid #23292c !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* Chip row: Streamlit's default vertical block stacks children top-to-
   bottom and stretches them full-width. Force it into a wrapping flex
   row of auto-sized, pill-shaped buttons instead, to match the chips in
   docs/index.html. */
.st-key-chip-row div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: .5rem !important;
    align-items: flex-start !important;
}
.st-key-chip-row div[data-testid="stVerticalBlock"] > div,
.st-key-chip-row div[data-testid="element-container"],
.st-key-chip-row div[data-testid="stButton"] {
    width: auto !important;
}
div.stButton > button {
    background: #1a2124; color: #d8dee2; border: 1px solid #23292c;
    border-radius: 999px; font-family: inherit; white-space: nowrap;
}
div.stButton > button:hover { border-color: #8a6238; color: #c98a4b; }

#output-frame {
    white-space: pre-wrap !important; word-break: break-word;
    line-height: 1.55; font-size: 0.92rem; color: #d8dee2;
    margin: 0 0 1rem; font-family: inherit;
}
#output-frame .prompt-echo { color: #6fcf97; }

.token-counter {
    color: #6fcf97; font-size: 0.82rem; margin: 0.75rem 0 0.4rem 0;
}

/* Prompt input styled like the promptline in docs/index.html: no visible
   box, blending into the footer instead of looking like a separate form
   field. The literal ">" caret is a column to its left (see layout below)
   rather than a CSS-positioned pseudo-element -- absolute positioning with
   a guessed pixel offset didn't line up correctly on a real device. */
.stTextInput input {
    background: transparent !important; color: #d8dee2 !important;
    border: none !important; border-bottom: 1px solid #23292c !important;
    border-radius: 0 !important; font-family: inherit !important;
}
.prompt-caret {
    color: #6fcf97; font-weight: 700; font-size: 1.1rem;
    display: flex; align-items: center; height: 2.6rem;
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

with st.sidebar:
    st.header("Parametros")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("Top-k", 1, 200, 50, 1)
    max_new_tokens = st.slider("Tokens a generar", 20, 400, 150, 10)
    st.markdown("---")
    st.markdown("[Codigo en GitHub](https://github.com/davidmorgadocarames/mini_llms)")
    # Bumped on every meaningful change, so a stale Streamlit Cloud deploy
    # (this has happened more than once) is instantly checkable instead of
    # guessed at: if this doesn't match the latest commit, it's a caching
    # issue on their end, not a bug in the code.
    st.caption("build: 2026-08-27.4 (real line breaks, not markdown-eaten)")

for key, default in (
    ("total_tokens", 0),
    ("last_output_prompt", ""),
    ("last_output_body", ""),
    ("last_processed", None),
    ("prompt", ""),
):
    if key not in st.session_state:
        st.session_state[key] = default


def render_output(prompt_text: str, body_text: str, cursor: bool = False) -> str:
    if not prompt_text and not body_text:
        return '<pre id="output-frame"></pre>'
    tail = "&#9608;" if cursor else ""
    # The model can emit the literal EOT separator token mid-generation when
    # it "jumps" to a new, unrelated article; show it as a paragraph break
    # instead of the raw <|endoftext|> string.
    body_display = clean_for_display(body_text)
    # st.markdown() runs its content through a Markdown parser even with
    # unsafe_allow_html=True, which collapses plain "\n\n" the way Markdown
    # normally treats whitespace -- confirmed on a real deploy: the
    # uppercase header transform worked but the line break silently
    # disappeared. Explicit <br> tags are immune to that, since Markdown
    # never touches text inside raw HTML tags it's already emitting.
    body_html = html.escape(body_display).replace("\n\n", "<br><br>")
    return (
        '<pre id="output-frame"><span class="prompt-echo" style="color:#6fcf97">'
        f"{html.escape(prompt_text)}</span> {body_html}{tail}</pre>"
    )


# --- fixed-height scrolling panel: example chips + generated output ---
st.markdown(
    '<div class="terminal-head-bar"><span class="dot"></span><span class="dot"></span>'
    '<span class="dot"></span>&nbsp;coconut &mdash; streamlit</div>',
    unsafe_allow_html=True,
)

clicked_prompt = None
with st.container(height=380, border=True):
    with st.container(key="chip-row"):
        for example in EXAMPLE_PROMPTS:
            if st.button(example, key=f"chip-{example}"):
                clicked_prompt = example
    output_box = st.empty()
    output_box.markdown(
        render_output(st.session_state.last_output_prompt, st.session_state.last_output_body),
        unsafe_allow_html=True,
    )

if clicked_prompt:
    st.session_state.prompt = clicked_prompt

# --- always-visible footer: token counter + prompt input, no button --
# Enter (or losing focus) submits, same as the promptline in docs/index.html.
token_counter = st.empty()


def render_token_counter(current: int = 0) -> None:
    session_total = st.session_state.total_tokens + current
    token_counter.markdown(
        f'<div class="token-counter">&#9679; tokens generados: {current} '
        f"(sesion: {session_total})</div>",
        unsafe_allow_html=True,
    )


render_token_counter()

caret_col, input_col = st.columns([0.03, 0.97])
with caret_col:
    st.markdown('<div class="prompt-caret">&gt;</div>', unsafe_allow_html=True)
with input_col:
    prompt = st.text_input(
        "Prompt", key="prompt", placeholder="elige un prompt de arriba, o escribelo tal cual",
        label_visibility="collapsed",
    )

to_generate = clicked_prompt or (prompt if prompt.strip() and prompt != st.session_state.last_processed else None)

if to_generate:
    ids = tokenizer.encode(to_generate)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = len(tokenizer.decode(ids))
    text_so_far = tokenizer.decode(ids)
    n_tokens = 0

    output_box.markdown(render_output(to_generate, "", cursor=True), unsafe_allow_html=True)

    for out_idx in model.generate_stream(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    ):
        text_so_far = tokenizer.decode(out_idx[0].tolist())
        n_tokens += 1
        output_box.markdown(
            render_output(to_generate, text_so_far[prompt_len:], cursor=True), unsafe_allow_html=True
        )
        render_token_counter(n_tokens)

    output_box.markdown(render_output(to_generate, text_so_far[prompt_len:]), unsafe_allow_html=True)
    st.session_state.last_output_prompt = to_generate
    st.session_state.last_output_body = text_so_far[prompt_len:]
    st.session_state.total_tokens += n_tokens
    st.session_state.last_processed = to_generate
    render_token_counter()
