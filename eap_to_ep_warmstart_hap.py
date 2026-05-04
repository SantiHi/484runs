"""
EAP -> Edge Pruning warm-start bridge (corrected version).

Uses the EP FPT2 model's built-in `add_or_remove_edge(from_node, to_node, value)`
API instead of reaching into log_alpha tensors directly. This is the
authoritative way to set per-edge log_alphas in this fork because it
handles writer indexing, head indexing, and the 2D vs 1D distinction
correctly for every reader type.

Pipeline:
  1. abs(score) per edge
  2. rank-normalize within each (reader) layer to [0, 1]
  3. fit logistic to hit target mean keep probability
  4. invert hard-concrete to get log_alpha
  5. call ep_model.add_or_remove_edge(from_node, to_node, value=log_alpha)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Hard-concrete L0 mask (Louizos 2018; same constants as the EP paper)
# ---------------------------------------------------------------------------
_BETA = 2.0 / 3.0
_L = -0.1
_R = 1.1
_EPS = 1e-6


def _hc_log_alpha_from_keep_prob(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    shift = _BETA * math.log(-_L / _R)
    return math.log(p / (1.0 - p)) + shift


# ---------------------------------------------------------------------------
# Translate EAP node names -> EP node names
# ---------------------------------------------------------------------------
def _eap_to_ep_writer_names(eap_parent: str) -> List[str]:
    """EAP parent -> list of EP writer names. Returns multiple for 'input'
    because EP has separate tok_embeds and pos_embeds writers but EAP has
    a single 'input' node. We attribute the score to both."""
    if eap_parent == "input":
        return ["tok_embeds", "pos_embeds"]
    if eap_parent.startswith("a"):    # 'a{L}.h{H}'
        return [eap_parent]
    if eap_parent.startswith("m"):    # 'm{L}'
        return [eap_parent]
    raise ValueError(f"Unknown EAP parent name: {eap_parent!r}")


def _eap_to_ep_reader_name(eap_child: str, edge_name: str) -> str:
    """EAP child + full edge name -> EP reader name.

    Q/K/V hint lives in the edge name suffix (e.g. 'input->a0.h0<q>'),
    not in the child name itself.
    """
    if eap_child == "logits":
        return "resid_post"
    if eap_child.startswith("m"):
        return eap_child
    if eap_child.startswith("a"):
        # Look for the qkv hint in the edge name
        for q in ("<q>", "<k>", "<v>"):
            if edge_name.endswith(q):
                return f"{eap_child}.{q[1]}"
        raise ValueError(
            f"Attention reader edge {edge_name!r} has no <q>/<k>/<v> hint"
        )
    raise ValueError(f"Unknown EAP child name: {eap_child!r}")


# ---------------------------------------------------------------------------
# Reader-layer extraction (for grouping)
# ---------------------------------------------------------------------------
def _ep_reader_layer(ep_reader_name: str, n_layers: int) -> int:
    if ep_reader_name == "resid_post":
        return n_layers
    if ep_reader_name.startswith("m"):
        return int(ep_reader_name[1:])
    if ep_reader_name.startswith("a"):
        # 'a{L}.h{H}.{q|k|v}'
        return int(ep_reader_name.split(".")[0][1:])
    raise ValueError(f"Unknown reader name: {ep_reader_name!r}")


# ---------------------------------------------------------------------------
# Per-layer rank normalization
# ---------------------------------------------------------------------------
def _rank_normalize_per_layer(
    edges: List[Tuple[str, str, float]],   # (writer, reader, score)
    layer_of: Callable[[str], int],         # reader_name -> layer
) -> List[float]:
    n = len(edges)
    layers = np.array([layer_of(r) for (_, r, _) in edges])
    scores = np.array([abs(s) for (_, _, s) in edges], dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)
    for L in np.unique(layers):
        idxs = np.where(layers == L)[0]
        sub = scores[idxs]
        order = np.argsort(sub, kind="stable")
        ranks = np.empty(len(sub), dtype=np.float64)
        ranks[order] = np.arange(len(sub))
        unique_vals, inv = np.unique(sub, return_inverse=True)
        for v_idx in range(len(unique_vals)):
            mask = inv == v_idx
            ranks[mask] = ranks[mask].mean()
        if len(sub) > 1:
            ranks /= (len(sub) - 1)
        else:
            ranks[:] = 0.5
        out[idxs] = ranks
    return out.tolist()


def _fit_logistic_to_target_keep(
    ranks: List[float], target_mean_keep: float, slope: float = 8.0,
) -> List[float]:
    r = np.array(ranks, dtype=np.float64)
    target = float(min(max(target_mean_keep, 1e-3), 1.0 - 1e-3))
    lo, hi = -5.0, 5.0
    for _ in range(60):
        bias = (lo + hi) / 2.0
        p = 1.0 / (1.0 + np.exp(-slope * (r - bias)))
        if p.mean() > target:
            lo = bias
        else:
            hi = bias
    bias = (lo + hi) / 2.0
    p = 1.0 / (1.0 + np.exp(-slope * (r - bias)))
    return p.tolist()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@dataclass
class WarmStartConfig:
    start_sparsity: float = 0.90
    logistic_slope: float = 8.0
    layer_grouping: str = "reader"   # only 'reader' supported here
    restrict_to_top_k_pct: Optional[float] = None  # None = no HAP restriction (original behavior)
    restricted_log_alpha: float = -10.0
    mask_save_path: Optional[str] = None


def warmstart_ep_from_eap(
    eap_graph,                # list of {name, parent, child, score} dicts
    ep_model,                 # FPT2LMHeadModel
    n_layers: int,
    n_heads: int,             # accepted for back-compat, not used
    config: WarmStartConfig = WarmStartConfig(),
    verbose: bool = True,
) -> Dict[str, float]:
    # Accept either a list of dicts or a Graph object
    if isinstance(eap_graph, list):
        edges_iter = eap_graph
    elif hasattr(eap_graph, "edges"):
        edges_iter = [
            {
                'name': name,
                'parent': e.parent.name,
                'child':  e.child.name,
                'score':  float(e.score) if e.score is not None else 0.0,
            }
            for name, e in eap_graph.edges.items()
        ]
    else:
        raise TypeError("eap_graph must be a list of dicts or a Graph object")

    # Translate every EAP edge to one or more (ep_writer, ep_reader, score)
    # tuples (one EAP edge from 'input' becomes two edges in EP, one from
    # tok_embeds and one from pos_embeds).
    triples: List[Tuple[str, str, float]] = []
    skipped = 0
    for edge in edges_iter:
        if edge.get('score') is None:
            skipped += 1
            continue
        try:
            ep_writers = _eap_to_ep_writer_names(edge['parent'])
            ep_reader = _eap_to_ep_reader_name(edge['child'], edge.get('name', ''))
        except (ValueError, KeyError):
            skipped += 1
            continue
        for w in ep_writers:
            triples.append((w, ep_reader, float(edge['score'])))

    if not triples:
        raise RuntimeError("No usable EAP edges. Did attribute() get called?")
    if verbose:
        print(f"[warmstart] harvested {len(triples)} EP edges from "
              f"{len(edges_iter)} EAP edges (skipped {skipped})")

    # Rank-normalize per reader-layer
    ranks = _rank_normalize_per_layer(
        triples,
        layer_of=lambda r: _ep_reader_layer(r, n_layers),
    )

    # Logistic fit
    target_keep = 1.0 - config.start_sparsity
    keep_probs = _fit_logistic_to_target_keep(
        ranks, target_keep, slope=config.logistic_slope,
    )
    if verbose:
        print(f"[warmstart] target mean keep = {target_keep:.3f}, "
              f"actual mean keep = {np.mean(keep_probs):.3f}")

    # Invert hard-concrete and write into the model via its native API
    log_alphas = [_hc_log_alpha_from_keep_prob(p) for p in keep_probs]

    n_written = 0
    n_failed = 0
    first_errors = []
    for (w, r, _), la in zip(triples, log_alphas):
        try:
            ep_model.add_or_remove_edge(w, r, value=la)
            n_written += 1
        except Exception as e:
            n_failed += 1
            if len(first_errors) < 5:
                first_errors.append(f"{w} -> {r}: {type(e).__name__}: {e}")

    if verbose:
        print(f"[warmstart] wrote {n_written} log_alphas, "
              f"failed on {n_failed}")
        for err in first_errors:
            print(f"  ! {err}")

    stats = {
        "n_eap_edges": float(len(edges_iter)),
        "n_ep_edges_translated": float(len(triples)),
        "n_skipped_at_parse": float(skipped),
        "n_log_alphas_written": float(n_written),
        "n_log_alphas_failed": float(n_failed),
        "mean_keep_prob": float(np.mean(keep_probs)),
        "min_keep_prob": float(np.min(keep_probs)),
        "max_keep_prob": float(np.max(keep_probs)),
        "mean_log_alpha": float(np.mean(log_alphas)),
    }

    # HAP-hard: mask out the bottom edges and build per-parameter freeze masks.
    if config.restrict_to_top_k_pct is not None:
        import torch
        from modeling_fpt2 import writer_name_to_idx

        n_total = len(triples)
        n_keep = int(n_total * config.restrict_to_top_k_pct)
        sorted_indices = sorted(
            range(n_total),
            key=lambda i: abs(triples[i][2]),
            reverse=True,
        )
        keep_set = set(sorted_indices[:n_keep])

        print(f"[HAP] keeping top {n_keep}/{n_total} edges "
              f"({config.restrict_to_top_k_pct * 100:.0f}%)")

        n_layers = ep_model.config.n_layer
        n_heads = ep_model.config.n_head
        masks = {}

        def _get_or_init_mask(param_name):
            if param_name not in masks:
                shape = ep_model.get_parameter(param_name).shape
                masks[param_name] = torch.zeros(shape, dtype=torch.bool)
            return masks[param_name]

        n_restricted = 0
        for i, (writer, reader, score) in enumerate(triples):
            if i in keep_set:
                continue
            # Force restricted log_alpha (overrides whatever main loop wrote)
            ep_model.add_or_remove_edge(writer, reader, value=config.restricted_log_alpha)

            from_idx = writer_name_to_idx(
                writer,
                num_layers=n_layers,
                num_heads=n_heads,
                with_embedding_nodes=True,
            )
            if reader == "resid_post":
                mask = _get_or_init_mask("transformer.final_read_log_alphas")
                mask[from_idx] = True
            elif reader.startswith("m"):
                layer_idx = int(reader[1:])
                mask = _get_or_init_mask(f"transformer.h.{layer_idx}.mlp_read_log_alphas")
                mask[from_idx] = True
            else:
                parts = reader.split(".")
                layer_idx = int(parts[0][1:])
                head_idx = int(parts[1][1:])
                qkv = parts[2]
                mask = _get_or_init_mask(
                    f"transformer.h.{layer_idx}.{qkv}_read_log_alphas"
                )
                mask[from_idx, head_idx] = True
            n_restricted += 1

        if config.mask_save_path:
            import os
            os.makedirs(os.path.dirname(config.mask_save_path), exist_ok=True)
            torch.save(masks, config.mask_save_path)
            n_frozen = sum(m.sum().item() for m in masks.values())
            print(f"[HAP] saved mask to {config.mask_save_path}: "
                  f"{n_frozen} entries frozen across {len(masks)} parameters")

        stats["n_restricted"] = n_restricted
        stats["n_kept"] = n_keep

    return stats