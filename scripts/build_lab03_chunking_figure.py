"""Figura de tamaño de fragmento y solapamiento para la sección 3 del laboratorio 3.

Usa el corte real de pol-equipaje con fragmentos de 200 tokens: #02 y #03 comparten una oración.
Se incluye en el paquete y el notebook la muestra con show_document; para regenerarla:
uv run --with matplotlib python scripts/build_lab03_chunking_figure.py src/curso_ia/data/lab03/chunking.png
"""

import sys
import textwrap

import matplotlib.pyplot as plt
from llama_index.core.node_parser import SentenceSplitter
from matplotlib.patches import FancyBboxPatch

from curso_ia.rag import load_corpus, sentences

CHUNK_SIZE, OVERLAP = 200, 20
COLORS = {"#02": "#4C78A8", "#03": "#F58518", "both": "#9D755D"}

documents, _ = load_corpus()
document = next(d for d in documents if d.metadata["doc_id"] == "pol-equipaje")
nodes = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP).get_nodes_from_documents([document])
first, second = nodes[1], nodes[2]
overlap = document.text[second.start_char_idx:first.end_char_idx]

# Oraciones de la política con su posición en el texto, desde el inicio de #02 hasta el final de #03.
rows, cursor = [], 0
for row in sentences([document]):
    start = document.text.index(row["sentence"], cursor)
    cursor = start + len(row["sentence"])
    if start >= first.start_char_idx and cursor <= second.end_char_idx:
        in_first, in_second = cursor <= first.end_char_idx, start >= second.start_char_idx
        rows.append((row["sentence"], "both" if in_first and in_second else "#02" if in_first else "#03"))

fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 1.6))
ax.set_xlim(0, 11)
ax.set_ylim(-len(rows) - 0.6, 1.2)
ax.axis("off")
for i, (sentence, owner) in enumerate(rows):
    y = -i - 0.5
    ax.add_patch(FancyBboxPatch((1.9, y - 0.4), 8.9, 0.8, boxstyle="round,pad=0.02",
                                facecolor=COLORS[owner], alpha=0.18 if owner != "both" else 0.35,
                                edgecolor=COLORS[owner], linewidth=1.2 if owner != "both" else 2.4))
    ax.text(2.0, y, textwrap.shorten(sentence, 92, placeholder=" …"), va="center", fontsize=9)
    if owner == "both":
        ax.text(10.7, y, "solapamiento", ha="right", va="center", fontsize=9.5, fontweight="bold",
                color=COLORS["both"])

def bracket(owner, label):
    ys = [-i - 0.5 for i, (_, o) in enumerate(rows) if o in (owner, "both")]
    x = 0.95 if owner == "#02" else 1.55
    ax.plot([x, x], [max(ys) + 0.4, min(ys) - 0.4], color=COLORS[owner], linewidth=4, solid_capstyle="butt")
    ax.text(x - 0.12, (max(ys) + min(ys)) / 2, label, rotation=90, ha="right", va="center",
            fontsize=10, color=COLORS[owner], fontweight="bold")

bracket("#02", "pol-equipaje#02")
bracket("#03", "pol-equipaje#03")
ax.text(0, 0.75, f"Fragmentos de hasta {CHUNK_SIZE} tokens con solapamiento de hasta {OVERLAP} tokens (10%)",
        fontsize=12, fontweight="bold")
ax.text(0, 0.3, f"La oración marcada queda en los dos fragmentos ({len(overlap)} caracteres compartidos). "
        "El corte nunca parte una oración.", fontsize=9.5, color="#444")
fig.savefig(sys.argv[1], dpi=110, bbox_inches="tight")
print("Solapamiento:", repr(overlap))
