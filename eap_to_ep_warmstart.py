"""
EAP -> Edge Pruning warm-start bridge.

Pipeline this implements:

    (1) Run EAP / EAP-IG in your existing notebook to get a scored
        eap.graph.Graph `g_eap` (edge.score is set for every edge).
    (2) Build an EP FPT2LMHeadModel `ep_model` (the GPT-2 variant from the
        princeton-nlp/Edge-Pruning repo, which adds per-edge log_alpha mask
        parameters on top of GPT-2).
    (3) Translate every EAP edge to its corresponding EP (writer, reader)
        index, take |score|, rank-normalize within each layer, fit a
        per-edge "keep probability" so the average matches a target start
        sparsity, invert the stretched-sigmoid hard-concrete expectation
        to recover log_alpha values, and write them into ep_model.
    (4) Hand `ep_model` to EP's standard training script
        (`src/prune/fpt2_ioi.py`) -- you only need to pass it a model that
        is already initialized, no other change to EP's pipeline.

This is the "sequential ensemble" / "warm-start edge pruning" idea
described in Mondorf et al. (BlackboxNLP 2025, "Exploring Ensemble
Strategies for Circuit Localization Methods"), reimplemented from
scratch.

================================================================
ASSUMPTIONS YOU MUST VERIFY ONCE FOR YOUR LIBRARY VERSIONS
================================================================
This module deliberately concentrates every external-library-internal
assumption in a small set of adapter functions at the top of the file
(`_eap_edge_endpoints`, `_set_log_alpha`, `_iter_ep_edge_keys`).

If something doesn't match your installed version, you will only need
to fix one of those adapters. Each one prints a clear error message
explaining what attribute it tried to read.

The assumptions are:

A1. eap.graph.Edge has a `parent` and `child` attribute, each of which
    has a `name` string. EAP node names follow the convention
        "input"                    (token embedding output, single node)
        "logits"                   (final logits node, single node)
        "a{layer}.h{head}"         (attention head output)
        "a{layer}.h{head}<q|k|v>"  (attention head Q/K/V input)
        "m{layer}"                 (MLP output / input pair share name)
    (This is the convention in hannamw/EAP-IG's graph.py as of 2024-2025.)

A2. The EP FPT2 model exposes its per-edge mask parameters via
        model.transformer.h[layer].q_read_log_alphas
        model.transformer.h[layer].k_read_log_alphas
        model.transformer.h[layer].v_read_log_alphas
        model.transformer.h[layer].mlp_read_log_alphas
        model.final_read_log_alphas  (or model.transformer.final_read_log_alphas)
    Each "read" tensor has shape `[n_writers]` where the writer index
    enumerates upstream nodes in a fixed canonical order:
        [embed, a0.h0, ..., a0.h{H-1}, m0, a1.h0, ..., m{L-1}]
    truncated to writers whose layer is <= the reader's layer.
    (This matches the FPT2 modeling convention; see
     princeton-nlp/Edge-Pruning/src/modeling/modeling_fpt2.py and
     vis_fpt2.py, which iterate writer indices in this order.)

If your install differs, edit ONLY the adapter functions below; the
rest of the file is library-agnostic math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Adapter A1: parse an EAP Edge into a (writer_kind, writer_layer, writer_head,
# reader_kind, reader_layer, reader_head_or_qkv) tuple.
# ---------------------------------------------------------------------------
# Tuple format used everywhere downstream:
#   writer kinds: 'embed' | 'attn' | 'mlp'
#   reader kinds: 'attn_q' | 'attn_k' | 'attn_v' | 'mlp' | 'logits'
# Layers/heads are int or None when irrelevant.

WriterKey = Tuple[str, Optional[int], Optional[int]]   # (kind, layer, head)
ReaderKey = Tuple[str, Optional[int], Optional[int]]   # (kind, layer, head)


def _parse_eap_node_name(name: str) -> Tuple[str, Optional[int], Optional[int], Optional[str]]:
    """Parse one EAP node name into (kind, layer, head, qkv_letter).
    qkv_letter is set only for attention input nodes ('a{L}.h{H}<q|k|v>').
    Raises ValueError on anything unrecognized -- callers should add a
    branch here if their EAP version uses a different naming convention.
    """
    if name == "input":
        return ("embed", None, None, None)
    if name == "logits":
        return ("logits", None, None, None)
    if name.startswith("m"):
        # 'm{layer}'
        return ("mlp", int(name[1:]), None, None)
    if name.startswith("a"):
        # 'a{L}.h{H}' or 'a{L}.h{H}<q|k|v>'
        # split on '.h'
        layer_str, rest = name[1:].split(".h", 1)
        layer = int(layer_str)
        if rest.endswith(("<q>", "<k>", "<v>")):
            head = int(rest[:-3])
            qkv = rest[-2]   # 'q', 'k', or 'v'
            return ("attn_in", layer, head, qkv)
        return ("attn_out", layer, int(rest), None)
    raise ValueError(
        f"Unrecognized EAP node name: {name!r}. "
        f"Edit _parse_eap_node_name to handle your library's convention."
    )


def _eap_edge_endpoints(edge) -> Tuple[WriterKey, ReaderKey]:
    """Adapter A1: return a (writer, reader) canonical tuple for an EAP edge.

    Reads `edge.parent.name` and `edge.child.name`. If your installed
    EAP-IG names them differently (e.g. `edge.upstream`, `edge.src`),
    edit just this function.
    """
    try:
        p_name = edge.parent.name
        c_name = edge.child.name
    except AttributeError as e:
        raise AttributeError(
            "EAP Edge does not expose `.parent.name` / `.child.name`. "
            "Inspect one edge with vars(edge) and patch _eap_edge_endpoints."
        ) from e

    p_kind, p_layer, p_head, _ = _parse_eap_node_name(p_name)
    c_kind, c_layer, c_head, c_qkv = _parse_eap_node_name(c_name)

    # Writer side: only embed / attn-out / mlp can be writers.
    if p_kind == "embed":
        writer: WriterKey = ("embed", None, None)
    elif p_kind == "attn_out":
        writer = ("attn", p_layer, p_head)
    elif p_kind == "mlp":
        writer = ("mlp", p_layer, None)
    else:
        raise ValueError(
            f"Edge parent {p_name!r} parsed to {p_kind}, which can't be a "
            f"writer in the EP model."
        )

    # Reader side: attn-Q/K/V inputs, MLP input, or final logits.
    if c_kind == "attn_in":
        reader: ReaderKey = (f"attn_{c_qkv}", c_layer, c_head)
    elif c_kind == "mlp":
        reader = ("mlp", c_layer, None)
    elif c_kind == "logits":
        reader = ("logits", None, None)
    else:
        raise ValueError(
            f"Edge child {c_name!r} parsed to {c_kind}, which can't be a "
            f"reader in the EP model."
        )

    return writer, reader


# ---------------------------------------------------------------------------
# Adapter A2: locate a specific log_alpha entry in the EP FPT2 model and
# write a value into it.
# ---------------------------------------------------------------------------
# The EP FPT2 model holds, for every reader R, a 1D parameter
#   reader.log_alphas of shape [n_writers_visible_to_R].
# Writer indices follow the canonical "writer order" defined below,
# truncated to writers with layer <= R's layer.

def _writer_index(writer: WriterKey, n_layers: int, n_heads: int) -> int:
    """Position of `writer` in the canonical full writer ordering:
        0:                embed
        1 + L*(H+1) + h:  a{L}.h{h}        (heads of layer L, h in 0..H-1)
        1 + L*(H+1) + H:  m{L}             (MLP of layer L)
    """
    kind, layer, head = writer
    if kind == "embed":
        return 0
    if kind == "attn":
        assert layer is not None and head is not None
        return 1 + layer * (n_heads + 1) + head
    if kind == "mlp":
        assert layer is not None
        return 1 + layer * (n_heads + 1) + n_heads
    raise ValueError(f"Bad writer kind: {kind}")


def _writer_layer(writer: WriterKey) -> int:
    """Layer index used for visibility comparison (embed = -1)."""
    kind, layer, _ = writer
    if kind == "embed":
        return -1
    assert layer is not None
    return layer


def _reader_layer(reader: ReaderKey, n_layers: int) -> int:
    """The layer below which writers are visible to this reader.
    For attn-Q/K/V at layer L: writers up to layer L-1 (and embed) are visible.
    For mlp at layer L:        writers up to layer L (incl. attn at L) are visible.
    For logits:                every writer is visible.
    """
    kind, layer, _ = reader
    if kind == "logits":
        return n_layers     # everything visible
    assert layer is not None
    if kind in ("attn_q", "attn_k", "attn_v"):
        # attn at layer L reads writers at strict-layer < L plus embed
        return layer - 1    # max visible writer-layer (embed has layer -1)
    if kind == "mlp":
        # mlp at L reads attn at L too -> max visible writer-layer = L
        return layer
    raise ValueError(f"Bad reader kind: {kind}")


def _resolve_reader_log_alpha(ep_model, reader: ReaderKey) -> torch.nn.Parameter:
    """Adapter A2: return the log_alpha tensor for a given reader.

    All access to EP model internals is here. If your fork of the EP
    code uses different attribute names, change them once below.
    """
    kind, layer, _ = reader
    try:
        h = ep_model.transformer.h
    except AttributeError as e:
        raise AttributeError(
            "EP model does not expose `.transformer.h`. Open the EP model "
            "in a debugger and patch _resolve_reader_log_alpha."
        ) from e

    if kind == "attn_q":
        return h[layer].q_read_log_alphas
    if kind == "attn_k":
        return h[layer].k_read_log_alphas
    if kind == "attn_v":
        return h[layer].v_read_log_alphas
    if kind == "mlp":
        return h[layer].mlp_read_log_alphas
    if kind == "logits":
        if hasattr(ep_model, 'final_read_log_alphas'):
            return ep_model.final_read_log_alphas
        if hasattr(ep_model.transformer, 'final_read_log_alphas'):
            return ep_model.transformer.final_read_log_alphas
        raise AttributeError("Can't find final_read_log_alphas on the model")
    raise ValueError(f"Unknown reader kind: {kind}")


def _set_log_alpha(
    ep_model,
    writer: WriterKey,
    reader: ReaderKey,
    value: float,
    n_layers: int,
    n_heads: int,
) -> None:
    """Write `value` into ep_model's log_alpha at (writer -> reader)."""
    tensor = _resolve_reader_log_alpha(ep_model, reader)
    full_idx = _writer_index(writer, n_layers, n_heads)
    # The reader's log_alpha is truncated to visible writers (writer-layer
    # <= reader's max writer-layer). `full_idx` already counts in that
    # canonical order, so as long as embed is index 0 and visible writers
    # appear before non-visible ones, this works.
    if full_idx >= tensor.numel():
        raise IndexError(
            f"Writer {writer} maps to full index {full_idx} but reader "
            f"{reader} only has a log_alpha of length {tensor.numel()}. "
            f"Either the writer-order constants don't match your EP "
            f"model, or this edge shouldn't exist (writer not visible "
            f"to reader). Inspect the model with vars(reader-module)."
        )
    with torch.no_grad():
        tensor[full_idx] = value


def _iter_ep_edge_keys(n_layers: int, n_heads: int) -> Iterable[Tuple[WriterKey, ReaderKey]]:
    """All (writer, reader) pairs that EP's FPT2 represents -- used only as
    a sanity check that every EP edge gets initialized.
    """
    # Writers (in canonical order)
    writers: List[WriterKey] = [("embed", None, None)]
    for L in range(n_layers):
        for h in range(n_heads):
            writers.append(("attn", L, h))
        writers.append(("mlp", L, None))

    # Readers
    readers: List[ReaderKey] = []
    for L in range(n_layers):
        for qkv in ("q", "k", "v"):
            for h in range(n_heads):
                readers.append((f"attn_{qkv}", L, h))
        readers.append(("mlp", L, None))
    readers.append(("logits", None, None))

    for r in readers:
        rl = _reader_layer(r, n_layers)
        for w in writers:
            if _writer_layer(w) <= rl:
                yield w, r


# ===========================================================================
# Library-agnostic math: scores -> keep probabilities -> log_alphas
# ===========================================================================
# Hard-concrete constants (Louizos 2018; matches the EP paper exactly).
_BETA = 2.0 / 3.0
_L = -0.1
_R = 1.1
_EPS = 1e-6


def _hc_keep_prob_from_log_alpha(log_alpha: float) -> float:
    """Probability that a hard-concrete sample is non-zero, given log_alpha.

    P(z != 0) = sigmoid(log_alpha - beta * log(-l/r)).
    """
    shift = _BETA * math.log(-_L / _R)
    return 1.0 / (1.0 + math.exp(-(log_alpha - shift)))


def _hc_log_alpha_from_keep_prob(p: float) -> float:
    """Inverse of the above: given a target keep probability p in (0, 1),
    return the log_alpha that produces it.
    """
    p = min(max(p, _EPS), 1.0 - _EPS)
    shift = _BETA * math.log(-_L / _R)
    return math.log(p / (1.0 - p)) + shift


# ---------------------------------------------------------------------------
# Per-layer rank-normalization
# ---------------------------------------------------------------------------
def _rank_normalize_per_layer(
    edges: List[Tuple[WriterKey, ReaderKey, float]],
    layer_of_edge: Callable[[WriterKey, ReaderKey], int],
) -> List[float]:
    """Group edges by layer, rank-normalize |score| within each group to
    [0, 1], return a list of normalized ranks aligned with `edges`.

    "Layer of an edge" is taken from the *reader*'s layer (this is what
    Mondorf et al. report). All edges sharing a reader layer compete for
    rank within that group.
    """
    n = len(edges)
    layers = np.array([layer_of_edge(w, r) for (w, r, _) in edges])
    scores = np.array([abs(s) for (_, _, s) in edges], dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)
    for L in np.unique(layers):
        idxs = np.where(layers == L)[0]
        sub = scores[idxs]
        # Average ranks (ties get the mean of their tied ranks).
        order = np.argsort(sub, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(sub))
        # Resolve ties to mean rank.
        unique_vals, inv = np.unique(sub, return_inverse=True)
        for v_idx in range(len(unique_vals)):
            mask = inv == v_idx
            ranks[mask] = ranks[mask].mean()
        if len(sub) > 1:
            ranks /= (len(sub) - 1)   # now in [0, 1]
        else:
            ranks[:] = 0.5
        out[idxs] = ranks
    return out.tolist()


# ---------------------------------------------------------------------------
# Logistic mapping rank -> keep probability, calibrated so the *mean* keep
# probability equals (1 - start_sparsity).
# ---------------------------------------------------------------------------
def _fit_logistic_to_target_keep(
    ranks: List[float],
    target_mean_keep: float,
    slope: float = 8.0,
    n_iter: int = 60,
) -> List[float]:
    """Map each rank r in [0, 1] to a keep probability via
        p(r) = sigmoid(slope * (r - bias))
    and binary-search the bias so that mean(p(r)) ~= target_mean_keep.

    Higher rank -> higher keep probability; high-attribution edges
    therefore start "more on". `slope` controls confidence: large values
    push toward {0, 1} initialization, small values keep things uncertain.
    """
    r = np.array(ranks, dtype=np.float64)
    target = float(min(max(target_mean_keep, 1e-3), 1.0 - 1e-3))

    lo, hi = -5.0, 5.0     # bias bounds (sigmoid arg roughly [-slope*5, slope*5])
    for _ in range(n_iter):
        bias = (lo + hi) / 2.0
        p = 1.0 / (1.0 + np.exp(-slope * (r - bias)))
        if p.mean() > target:
            lo = bias    # need to lower mean -> raise bias
        else:
            hi = bias
    bias = (lo + hi) / 2.0
    p = 1.0 / (1.0 + np.exp(-slope * (r - bias)))
    return p.tolist()


# ===========================================================================
# Top-level entry point
# ===========================================================================
@dataclass
class WarmStartConfig:
    """Configuration for the EAP -> EP warm-start conversion.

    start_sparsity:
        The fraction of edges you want to be *off* on average at the
        start of EP training. Mondorf et al. sweep {0.80, 0.90, 0.95}.
        Should match (or be slightly looser than) the
        --start_edge_sparsity flag you'll pass to fpt2_ioi.py.

    logistic_slope:
        Controls how aggressively rank determines keep probability.
        Slope=8 is a reasonable default; lower values (~3-4) leave more
        room for EP to override EAP's ordering, higher values (~12-16)
        commit more strongly to it.

    layer_grouping:
        'reader'  -> rank-normalize within each reader layer (paper recipe).
        'writer'  -> rank-normalize within each writer layer.
        'global'  -> single global ranking, no per-layer normalization.

        The paper uses per-layer; we default to 'reader' since EP's
        gradient signal also flows backward from the reader.
    """
    start_sparsity: float = 0.90
    logistic_slope: float = 8.0
    layer_grouping: str = "reader"


def warmstart_ep_from_eap(
    eap_graph,
    ep_model,
    n_layers: int,
    n_heads: int,
    config: WarmStartConfig = WarmStartConfig(),
    verbose: bool = True,
) -> Dict[str, float]:
    """Translate EAP/-IG scores on `eap_graph` into log_alpha values on
    `ep_model`. Returns a dict of summary statistics.

    Order of operations exactly matches Mondorf et al. (BlackboxNLP 2025):
      1. abs(score) per edge
      2. rank-normalize within each (reader) layer to [0, 1]
      3. fit logistic to hit target mean keep probability
      4. invert hard-concrete to get log_alpha

    Edges in EP's graph that have no corresponding EAP edge keep
    whatever value was on the model already (typically EP's default
    init from `reset_all_log_alphas()` -- call that BEFORE this function
    if you want a clean slate).
    """
    # ---- step 0: harvest (writer, reader, score) tuples from EAP graph ---
    # Accept either a Graph object (with .edges) or a list of dicts.
    if isinstance(eap_graph, list):
        edge_iter = eap_graph
        get_score = lambda e: e.get('score')
        get_parent = lambda e: e['parent']
        get_child  = lambda e: e['child']
    elif hasattr(eap_graph, "edges"):
        edge_iter = eap_graph.edges.values()
        get_score = lambda e: getattr(e, 'score', None)
        get_parent = lambda e: e.parent.name
        get_child  = lambda e: e.child.name
    else:
        raise AttributeError(
            "eap_graph must be either a list of {'parent','child','score'} "
            "dicts or an object with `.edges`."
        )

    triples: List[Tuple[WriterKey, ReaderKey, float]] = []
    skipped = 0
    for edge in edge_iter:
        score = get_score(edge)
        if score is None:
            skipped += 1
            continue
        try:
            p_kind, p_layer, p_head, _ = _parse_eap_node_name(get_parent(edge))
            child_name = get_child(edge)
            edge_name = edge.get('name', '') if isinstance(edge, dict) else ''
            qkv_hint = None
            for q in ('<q>', '<k>', '<v>'):
                if edge_name.endswith(q):
                    qkv_hint = q[1]
                    break
            if qkv_hint is not None and not child_name.endswith(('<q>', '<k>', '<v>')):
                child_name_for_parse = f"{child_name}<{qkv_hint}>"
            else:
                child_name_for_parse = child_name
            c_kind, c_layer, c_head, c_qkv = _parse_eap_node_name(child_name_for_parse)
            # writer side
            if p_kind == "embed":
                w: WriterKey = ("embed", None, None)
            elif p_kind == "attn_out":
                w = ("attn", p_layer, p_head)
            elif p_kind == "mlp":
                w = ("mlp", p_layer, None)
            else:
                raise ValueError(f"bad parent kind {p_kind}")
            # reader side
            if c_kind == "attn_in":
                r: ReaderKey = (f"attn_{c_qkv}", c_layer, c_head)
            elif c_kind == "mlp":
                r = ("mlp", c_layer, None)
            elif c_kind == "logits":
                r = ("logits", None, None)
            else:
                raise ValueError(f"bad child kind {c_kind}")
        except (ValueError, AttributeError, KeyError):
            skipped += 1
            continue
        triples.append((w, r, float(score)))

    if not triples:
        raise RuntimeError(
            "No EAP edges had usable scores. Did you actually call "
            "attribute(...) on the graph before warmstarting?"
        )

    if verbose:
        print(f"[warmstart] harvested {len(triples)} EAP edges "
              f"(skipped {skipped})")

    # ---- step 1+2: rank-normalize per (reader) layer --------------------
    if config.layer_grouping == "reader":
        layer_of = lambda w, r: _reader_layer(r, n_layers)
    elif config.layer_grouping == "writer":
        layer_of = lambda w, r: _writer_layer(w)
    elif config.layer_grouping == "global":
        layer_of = lambda w, r: 0
    else:
        raise ValueError(f"Bad layer_grouping: {config.layer_grouping}")

    ranks = _rank_normalize_per_layer(triples, layer_of)

    # ---- step 3: logistic fit to target mean keep probability ----------
    target_keep = 1.0 - config.start_sparsity
    keep_probs = _fit_logistic_to_target_keep(
        ranks, target_keep, slope=config.logistic_slope,
    )

    if verbose:
        print(f"[warmstart] target mean keep = {target_keep:.3f}, "
              f"actual mean keep = {np.mean(keep_probs):.3f}")

    # ---- step 4: invert hard-concrete, write into ep_model -------------
    log_alphas = [_hc_log_alpha_from_keep_prob(p) for p in keep_probs]

    n_written = 0
    n_failed = 0
    for (w, r, _), la in zip(triples, log_alphas):
        try:
            _set_log_alpha(ep_model, w, r, la, n_layers, n_heads)
            n_written += 1
        except (IndexError, AttributeError, ValueError) as e:
            n_failed += 1
            if verbose and n_failed <= 5:
                print(f"[warmstart] failed to set ({w} -> {r}): {e}")

    if verbose:
        print(f"[warmstart] wrote {n_written} log_alphas, "
              f"failed on {n_failed}")

    return {
        "n_eap_edges": float(len(triples)),
        "n_skipped_at_parse": float(skipped),
        "n_log_alphas_written": float(n_written),
        "n_log_alphas_failed": float(n_failed),
        "mean_keep_prob": float(np.mean(keep_probs)),
        "min_keep_prob": float(np.min(keep_probs)),
        "max_keep_prob": float(np.max(keep_probs)),
        "mean_log_alpha": float(np.mean(log_alphas)),
    }
