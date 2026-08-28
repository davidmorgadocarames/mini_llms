# Archived: results affected by the Sliced decoder-mask bug

These C.6 results (GSM8K accuracy-by-steps, efficiency, and the 141-example
domain eval set) were produced with a Sliced checkpoint trained under a real
bug: `coconut_lab/models/sliced.py::train_steps` masked the decoder's
position-0 BOS token as if it were padding (BOS and pad share the same
token id in this project's BPE vocab), leaving the first self-attention
query with zero valid keys under causal masking. Sliced could never learn
to predict a response's first token from real context -- every generation
started broken (e.g. "ue........." instead of "blue.").

Fixed in commit `2e36ced` (`tgt_pad_mask[:, 0] = False`). Sliced was
retrained from scratch with the fix and all three affected C.6 pieces were
re-run; the active results in `coconut_lab/eval/results/` are the corrected
ones. These files are kept only as a record of the bug's real impact, not
as a valid comparison baseline.
