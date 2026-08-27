"""Fase B interactive demo: type or generate a nested boolean expression at
any depth and watch all three architectures evaluate it -- including the
Looped Locate-and-Replace pipeline's step-by-step reduction, animated one
step at a time. Same Streamlit app as the Fase A demo (streamlit_app.py),
just a second page -- Streamlit auto-discovers pages/*.py and adds the page
picker to the sidebar, no extra deployment needed.

Model weights are the ones depth_lab/eval/run_eval.py already trained and
evaluated (see the accuracy-vs-depth chart in the README); they're hosted
alongside the Fase A checkpoint on the same HuggingFace Hub model repo,
under a depth_lab/ prefix.
"""

import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import streamlit as st
from depth_lab.data.generator import generate
from depth_lab.data.reduce import evaluate as reduce_evaluate
from depth_lab.models import baseline as baseline_mod
from depth_lab.models import encoder_decoder as encdec_mod
from depth_lab.models import llr_loop
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.replacer import Replacer, ReplacerConfig
from depth_lab.tokenizer import CharTokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Fase B - Depth Lab", page_icon="\U0001f300", layout="wide", initial_sidebar_state="collapsed"
)

PAGE_CSS = """
<style>
.stApp { background-color: #0b0e0f; }
html, body, [class*="css"], .stApp, .stMarkdown, .stButton button, .stTextInput input {
    font-family: ui-monospace, SFMono-Regular, "Cascadia Code", "Fira Code",
                 Consolas, "Courier New", monospace !important;
}
.stApp, .stApp p, .stApp label, .stApp span { color: #d8dee2; }
[data-testid="stSidebar"] { background-color: #12171a; border-right: 1px solid #23292c; }
[data-testid="stHeader"] { background: #0b0e0f; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; }
.block-container { max-width: 1100px; margin: 0 auto; }

.depthlab-expr { font-size: 1.1rem; padding: 0.6rem 0.8rem; background: #12171a;
    border: 1px solid #23292c; border-radius: 6px; margin: 0.5rem 0 1rem; }
.depthlab-highlight { background: #8a6238; color: #0b0e0f; border-radius: 3px; padding: 0 2px; }
.depthlab-ok { color: #6fcf97; }
.depthlab-bad { color: #eb5757; }
.depthlab-step { color: #7b8790; font-size: 0.9rem; margin: 0.15rem 0; }
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Descargando los modelos de Fase B (solo la primera vez)...")
def load_models():
    tokenizer = CharTokenizer()

    def _load(filename, config_cls, model_cls):
        path = hf_hub_download(HF_REPO, f"depth_lab/{filename}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = model_cls(config_cls(**ckpt["config"])).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    baseline_model = _load("baseline_bool.pt", baseline_mod.GPTConfig, baseline_mod.GPT)
    encdec_model = _load("encdec_bool.pt", encdec_mod.EncDecConfig, encdec_mod.EncoderDecoderTransformer)
    locator = _load("locator_bool.pt", LocatorConfig, Locator)
    replacer = _load("replacer_bool.pt", ReplacerConfig, Replacer)
    return tokenizer, baseline_model, encdec_model, locator, replacer


tokenizer, baseline_model, encdec_model, locator, replacer = load_models()

st.title("Fase B — Depth Lab")
st.markdown(
    "Un Transformer entrenado para evaluar expresiones anidadas booleanas "
    "(`(True and (False or not (True)))`) aprende perfectamente hasta la profundidad "
    "que vio en entrenamiento (0-5) ... y luego falla cada vez mas cuanto mas profunda es la "
    "expresion, aunque sea mas corta que otras que si resuelve bien. Prueba una expresion "
    "de profundidad alta (8-12, fuera de lo visto en entrenamiento) y compara las 3 arquitecturas."
)

with st.sidebar:
    st.header("Expresion")
    depth = st.slider("Profundidad", 0, 14, 10)
    if st.button("Generar expresion aleatoria"):
        st.session_state["depthlab_expr"] = generate("bool", depth, max_shallow=2).expr
    st.markdown("---")
    st.markdown("[Codigo en GitHub](https://github.com/davidmorgadocarames/mini_llms)")
    st.caption(f"corriendo en `{DEVICE}`")

if "depthlab_expr" not in st.session_state:
    st.session_state["depthlab_expr"] = generate("bool", depth, max_shallow=2).expr

expr = st.text_input("Expresion booleana", key="depthlab_expr")
st.markdown(f'<div class="depthlab-expr">{expr}</div>', unsafe_allow_html=True)

evaluate_clicked = st.button("Evaluar con las 3 arquitecturas", type="primary")

if evaluate_clicked:
    try:
        tokenizer.encode(expr)
        true_value = str(reduce_evaluate(expr))
    except (ValueError, KeyError, IndexError):
        st.error(
            "No se pudo interpretar la expresion. Usa solo True/False, and/or/not, "
            "parentesis, y asegurate de que los parentesis estan balanceados."
        )
        st.stop()

    st.markdown(f"**Valor real:** `{true_value}`")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Decoder-only")
        pred = baseline_mod.predict_one(baseline_model, tokenizer, expr, DEVICE)
        css = "depthlab-ok" if pred == true_value else "depthlab-bad"
        st.markdown(f'<span class="{css}">{pred or "(vacio)"}</span>', unsafe_allow_html=True)

    with col2:
        st.subheader("Encoder-decoder")
        try:
            pred = encdec_mod.predict_one(encdec_model, tokenizer, expr, DEVICE)
        except ValueError:
            pred = "(expresion demasiado larga)"
        css = "depthlab-ok" if pred == true_value else "depthlab-bad"
        st.markdown(f'<span class="{css}">{pred or "(vacio)"}</span>', unsafe_allow_html=True)

    with col3:
        st.subheader("Looped Locate-and-Replace")
        trace_box = st.empty()
        result = llr_loop.reduce_with_llr(locator, replacer, tokenizer, expr, DEVICE)

        rendered_steps = []
        for i, step in enumerate(result.steps):
            start, end = step.span
            highlighted = (
                step.expr[:start]
                + f'<span class="depthlab-highlight">{step.expr[start:end]}</span>'
                + step.expr[end:]
            )
            rendered_steps.append(
                f'<div class="depthlab-step">paso {i + 1}: {highlighted} &rarr; {step.predicted_value}</div>'
            )
            trace_box.markdown("".join(rendered_steps), unsafe_allow_html=True)
            time.sleep(0.4)

        final = result.final_expr if result.converged else f"{result.final_expr} (no convergio)"
        css = "depthlab-ok" if result.converged and result.final_expr == true_value else "depthlab-bad"
        st.markdown(f'<span class="{css}">{final}</span>', unsafe_allow_html=True)
