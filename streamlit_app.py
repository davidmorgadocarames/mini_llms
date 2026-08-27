"""Streamlit Community Cloud app: real, live inference with the trained
Fase A checkpoint (downloaded from the HuggingFace Hub model repo
davidmorgado/coconut-mini-llm at startup, then cached).

This is the "real inference in the browser" counterpart to the static replay
demo at docs/index.html — here the model actually runs, on Streamlit's
server, not a pre-recorded transcript.
"""

import torch
from huggingface_hub import hf_hub_download

import streamlit as st
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"

EXAMPLE_PROMPTS = [
    "The history of the",
    "In 1943, the",
    "The film received",
    "The album was praised by critics for its",
]

st.set_page_config(page_title="Coconut mini-LLM", page_icon="🥥")


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

st.title("🥥 Coconut")
st.caption(
    f"Mini-LLM de {model.num_parameters() / 1e6:.1f}M parametros (RoPE, RMSNorm, SwiGLU, "
    f"Grouped Query Attention), entrenado desde cero sobre WikiText-103 · step {step:,} · "
    f"corriendo en `{device}`."
)
st.info(
    "Coconut es un **modelo base** (sin fine-tuning de instrucciones): escribe el "
    "principio de una frase para que la continue, no le hagas preguntas directas "
    "-- funciona mejor con prosa tipo Wikipedia que con conversacion.",
    icon="ℹ️",
)

with st.sidebar:
    st.header("Parametros de generacion")
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

if st.button("Generar", type="primary") and prompt.strip():
    def token_stream():
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        prev_text = tokenizer.decode(ids)
        yield prev_text
        for out_idx in model.generate_stream(
            idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        ):
            full_text = tokenizer.decode(out_idx[0].tolist())
            yield full_text[len(prev_text):]
            prev_text = full_text

    st.markdown("**Salida:**")
    st.write_stream(token_stream())
