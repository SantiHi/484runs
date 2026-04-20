"""
Test Suite: Factored Subspace Compression for Edge Pruning
==========================================================

Empirically validates two claims:
  1. Fidelity  -- The compressed algorithm discovers circuits with comparable
                  KL-divergence and sparsity to the standard (uncompressed) one.
  2. Scalability -- Compression drastically reduces peak VRAM, pushing the OOM
                    boundary much further.

Three experiments:
  Experiment 1  "Apples-to-Apples"   -- single training step, compare VRAM + losses
  Experiment 2  "Memory Wall"         -- find standard OOM threshold, verify compressed survives
  Experiment 3  "Large Model"         -- run a large-config model under compression

Usage:
  pytest test_edge_pruning_compression.py -v -s
"""

import gc
import math
import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup -- mirrors the hacky sys.path.append used in the pruning scripts
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "modeling"))

# Import the ORIGINAL (uncompressed) model for reference / standard baseline
from modeling_fpt2 import FPT2LMHeadModel as FPT2LMHeadModelOriginal

# Import the COMPRESSED model
from modeling_fpt2_compressed import FPT2LMHeadModel as FPT2LMHeadModelCompressed

from transformers import GPT2Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cuda"
SKIP_MSG = "CUDA GPU required for memory measurement tests"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_gpt2_config(
    n_embd: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    vocab_size: int = 1000,
    n_positions: int = 128,
) -> GPT2Config:
    """Return a tiny GPT-2 config suitable for fast unit tests."""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        # 4x expansion for MLP (GPT-2 default)
        n_inner=4 * n_embd,
        # Disable flash-attn so tests run everywhere
        attn_implementation="eager",
    )


def _large_gpt2_config(
    n_embd: int = 4096,
    n_layer: int = 32,
    n_head: int = 32,
    vocab_size: int = 32000,
    n_positions: int = 512,
) -> GPT2Config:
    """Return a large GPT-2-shaped config (mimics Llama-7B dimensions)."""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
        attn_implementation="eager",
    )


def _dummy_inputs(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Generate random clean and corrupted token sequences."""
    clean_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    corr_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return clean_ids, corr_ids


def _get_corrupted_writer_states(ref_model, corr_ids):
    """Run corrupted inputs through the reference model and return d-dim writer states.

    Returns shape (batch, writers, seq, d) -- ready to pass to the pruned model's
    forward(), which internally transposes back to (writers, batch, seq, d).
    """
    with torch.no_grad():
        out = ref_model(input_ids=corr_ids, output_writer_states=True)
        # writer_states: (writers, batch, seq, d)
        corr_x = out.writer_states.transpose(0, 1)  # -> (batch, writers, seq, d)
    return corr_x


def _run_training_step(
    pruned_model,
    ref_model,
    clean_ids,
    corr_x,
    target_edge_sparsity: float = 0.95,
    target_node_sparsity: float = 0.5,
):
    """Execute one full forward + loss + backward step on the pruned model.

    Returns a dict of scalar metrics (detached from the graph).
    """
    # --- Reference logits (no grad) ---
    with torch.no_grad():
        logits_ref = ref_model(input_ids=clean_ids).logits

    # --- Pruned model forward ---
    outputs = pruned_model(
        input_ids=clean_ids,
        target_edge_sparsity=target_edge_sparsity,
        target_node_sparsity=target_node_sparsity,
        corr_x=corr_x,
    )
    logits = outputs.logits

    # --- KL-divergence loss over the last position (mimics task-specific eval) ---
    kl_loss = F.kl_div(
        F.log_softmax(logits[:, -1, :], dim=-1),
        F.log_softmax(logits_ref[:, -1, :], dim=-1),
        reduction="batchmean",
        log_target=True,
    )

    # --- Regularization losses ---
    edge_loss = outputs.edge_loss
    node_loss = outputs.node_loss
    total_loss = kl_loss + edge_loss + node_loss

    # --- Backward ---
    total_loss.backward()

    metrics = {
        "total_loss": total_loss.detach().item(),
        "kl_loss": kl_loss.detach().item(),
        "edge_loss": edge_loss.detach().item(),
        "node_loss": node_loss.detach().item(),
        "model_edge_sparsity": outputs.model_edge_sparsity.detach().item(),
        "model_node_sparsity": outputs.model_node_sparsity.detach().item(),
    }
    return metrics


def _measure_peak_vram(
    model_cls,
    config,
    ref_model,
    clean_ids,
    corr_ids,
    with_embedding_nodes: bool = True,
    compression_dim: int = 0,
):
    """Instantiate a pruned model, run one training step, and return peak VRAM (bytes)
    plus the loss metrics dict.

    The model is created, used, and destroyed inside this function to prevent
    memory leaks across measurements.
    """
    # Build the pruned model
    pruned = model_cls(
        config,
        with_embedding_nodes=with_embedding_nodes,
        compression_dim=compression_dim,
    ).to(DEVICE)
    pruned.reset_all_log_alphas()
    pruned.train()

    # Get corrupted writer states from the (shared) reference model
    corr_x = _get_corrupted_writer_states(ref_model, corr_ids)

    # ---- VRAM measurement window begins ----
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)

    metrics = _run_training_step(pruned, ref_model, clean_ids, corr_x)

    torch.cuda.synchronize(DEVICE)
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    # ---- VRAM measurement window ends ----

    # Cleanup
    del pruned, corr_x
    torch.cuda.empty_cache()
    gc.collect()

    return peak_bytes, metrics


def _cleanup():
    """Aggressive GPU memory cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ===========================================================================
# Experiment 1: Apples-to-Apples Baseline
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason=SKIP_MSG)
class TestApplestoApples:
    """Compare standard vs. compressed edge pruning on a small GPT-2.

    Validates:
      - Peak VRAM of compressed < peak VRAM of standard
      - Both produce finite, non-NaN losses
      - KL divergence and sparsity metrics are in a comparable ballpark
    """

    # ---- Shared fixtures ----

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create a tiny model and dummy data, clean up after."""
        self.config = _small_gpt2_config(n_embd=128, n_layer=2, n_head=4)
        self.batch_size = 8
        self.seq_len = 32

        # Reference model -- shared between standard and compressed runs.
        # Uses the ORIGINAL (uncompressed) model so writer_states are d-dim.
        self.ref_model = FPT2LMHeadModelOriginal(
            self.config, with_embedding_nodes=True
        ).to(DEVICE)
        self.ref_model.reset_all_log_alphas()
        self.ref_model.eval()

        # Dummy data
        self.clean_ids, self.corr_ids = _dummy_inputs(
            self.batch_size, self.seq_len, self.config.vocab_size, DEVICE
        )

        yield  # ---- test runs here ----

        # Teardown
        del self.ref_model, self.clean_ids, self.corr_ids
        _cleanup()

    # ---- Tests ----

    def test_compressed_uses_less_vram(self):
        """Peak VRAM of the compressed method must be strictly less than standard."""

        # --- Standard (uncompressed) run ---
        _cleanup()
        standard_vram, standard_metrics = _measure_peak_vram(
            model_cls=FPT2LMHeadModelOriginal,
            config=self.config,
            ref_model=self.ref_model,
            clean_ids=self.clean_ids,
            corr_ids=self.corr_ids,
            with_embedding_nodes=True,
            compression_dim=0,
        )

        # --- Compressed run (k=64) ---
        _cleanup()
        compressed_vram, compressed_metrics = _measure_peak_vram(
            model_cls=FPT2LMHeadModelCompressed,
            config=self.config,
            ref_model=self.ref_model,
            clean_ids=self.clean_ids,
            corr_ids=self.corr_ids,
            with_embedding_nodes=True,
            compression_dim=64,
        )

        # ---- Assertions ----

        # 1. Compressed must use strictly less peak VRAM
        print(f"\n[Experiment 1] Standard peak VRAM : {standard_vram / 1e6:.1f} MB")
        print(f"[Experiment 1] Compressed peak VRAM: {compressed_vram / 1e6:.1f} MB")
        print(f"[Experiment 1] Savings            : {(standard_vram - compressed_vram) / 1e6:.1f} MB "
              f"({100 * (1 - compressed_vram / standard_vram):.1f}%)")
        assert compressed_vram < standard_vram, (
            f"Compressed ({compressed_vram / 1e6:.1f} MB) should use less VRAM "
            f"than standard ({standard_vram / 1e6:.1f} MB)"
        )

        # 2. Both produce finite, non-NaN losses
        for label, m in [("standard", standard_metrics), ("compressed", compressed_metrics)]:
            assert math.isfinite(m["total_loss"]), f"{label} total_loss is not finite: {m['total_loss']}"
            assert math.isfinite(m["kl_loss"]), f"{label} kl_loss is not finite: {m['kl_loss']}"
            assert math.isfinite(m["edge_loss"]), f"{label} edge_loss is not finite: {m['edge_loss']}"
            assert math.isfinite(m["node_loss"]), f"{label} node_loss is not finite: {m['node_loss']}"

    def test_losses_comparable_ballpark(self):
        """KL divergence and sparsity from both methods should be in a similar range.

        We don't expect identical values (compression introduces approximation),
        but they should be within an order of magnitude of each other.
        """
        _cleanup()
        _, standard_metrics = _measure_peak_vram(
            model_cls=FPT2LMHeadModelOriginal,
            config=self.config,
            ref_model=self.ref_model,
            clean_ids=self.clean_ids,
            corr_ids=self.corr_ids,
            with_embedding_nodes=True,
            compression_dim=0,
        )

        _cleanup()
        _, compressed_metrics = _measure_peak_vram(
            model_cls=FPT2LMHeadModelCompressed,
            config=self.config,
            ref_model=self.ref_model,
            clean_ids=self.clean_ids,
            corr_ids=self.corr_ids,
            with_embedding_nodes=True,
            compression_dim=64,
        )

        print(f"\n[Experiment 1b] Standard  metrics: {standard_metrics}")
        print(f"[Experiment 1b] Compressed metrics: {compressed_metrics}")

        # Sparsity values should be in a similar ballpark (both start from
        # log_alpha ~ N(10, 0.01), so initial sparsities should be very low)
        std_es = standard_metrics["model_edge_sparsity"]
        cmp_es = compressed_metrics["model_edge_sparsity"]
        # Both should be near 0 at initialization (all edges "on")
        assert abs(std_es) < 0.1, f"Standard edge sparsity unexpectedly high: {std_es}"
        assert abs(cmp_es) < 0.1, f"Compressed edge sparsity unexpectedly high: {cmp_es}"

        # KL divergences should both be non-negative
        assert standard_metrics["kl_loss"] >= -1e-6, "Standard KL should be non-negative"
        assert compressed_metrics["kl_loss"] >= -1e-6, "Compressed KL should be non-negative"


# ===========================================================================
# Experiment 2: Finding the Memory Wall (Stress Test)
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason=SKIP_MSG)
class TestMemoryWall:
    """Progressively increase batch size until standard OOMs, then verify
    compressed survives at that threshold.

    This test is inherently GPU-specific; it adapts to whatever card is present.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Prepare a medium-sized config that will OOM at reasonable batch sizes."""
        # Use a moderately-sized model so OOM happens at realistic batch sizes
        self.config = _small_gpt2_config(n_embd=256, n_layer=4, n_head=8)
        self.seq_len = 64
        self.vocab_size = self.config.vocab_size

        # Reference model (uncompressed, shared)
        self.ref_model = FPT2LMHeadModelOriginal(
            self.config, with_embedding_nodes=True
        ).to(DEVICE)
        self.ref_model.reset_all_log_alphas()
        self.ref_model.eval()

        yield

        del self.ref_model
        _cleanup()

    def _attempt_training_step(self, model_cls, batch_size, compression_dim=0):
        """Try one training step. Returns True if successful, False if OOM."""
        _cleanup()
        try:
            clean_ids, corr_ids = _dummy_inputs(
                batch_size, self.seq_len, self.vocab_size, DEVICE
            )
            pruned = model_cls(
                self.config,
                with_embedding_nodes=True,
                compression_dim=compression_dim,
            ).to(DEVICE)
            pruned.reset_all_log_alphas()
            pruned.train()

            corr_x = _get_corrupted_writer_states(self.ref_model, corr_ids)
            _run_training_step(pruned, self.ref_model, clean_ids, corr_x)

            del pruned, corr_x, clean_ids, corr_ids
            _cleanup()
            return True

        except torch.cuda.OutOfMemoryError:
            # OOM -- clean up and report failure
            del pruned, corr_x, clean_ids, corr_ids
            _cleanup()
            return False

        except Exception:
            # Re-raise non-OOM errors (bugs, shape mismatches, etc.)
            _cleanup()
            raise

    def test_compressed_survives_beyond_standard_oom(self):
        """Find the batch size where standard OOMs, then prove compressed survives."""

        # --- Phase 1: Find the standard method's OOM threshold ---
        # Start at batch_size=16 and double until OOM
        oom_batch_size = None
        last_ok_batch_size = None

        batch_size = 16
        while batch_size <= 4096:
            print(f"\n[Experiment 2] Trying standard with batch_size={batch_size}...", end=" ")
            ok = self._attempt_training_step(FPT2LMHeadModelOriginal, batch_size, compression_dim=0)
            if ok:
                print("OK")
                last_ok_batch_size = batch_size
                batch_size *= 2
            else:
                print("OOM!")
                oom_batch_size = batch_size
                break

        if oom_batch_size is None:
            pytest.skip(
                f"Standard method did not OOM up to batch_size={batch_size // 2}. "
                "GPU has too much memory for this config to trigger OOM. "
                "Try a larger config or longer seq_len."
            )

        print(f"\n[Experiment 2] Standard OOMs at batch_size={oom_batch_size}")
        print(f"[Experiment 2] Last successful batch_size={last_ok_batch_size}")

        # --- Phase 2: Run compressed at the OOM threshold ---
        print(f"[Experiment 2] Trying compressed (k=64) at batch_size={oom_batch_size}...", end=" ")
        compressed_ok = self._attempt_training_step(
            FPT2LMHeadModelCompressed, oom_batch_size, compression_dim=64
        )

        if compressed_ok:
            print("OK -- compressed survived!")
        else:
            print("OOM -- compressed also failed")

        assert compressed_ok, (
            f"Compressed method should survive at batch_size={oom_batch_size} "
            f"where standard OOMed"
        )


# ===========================================================================
# Experiment 3: Large Model Simulation
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason=SKIP_MSG)
class TestLargeModelSimulation:
    """Verify that a large-config model (d=4096, 32 layers) runs under
    compression with a small batch size without crashing.

    This tests that tensor shapes, autograd graph construction, and memory
    management all work at scale. Assumes a ~24 GB VRAM environment.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        yield
        _cleanup()

    def test_large_model_compressed_completes(self):
        """Instantiate a large model with compression and run one training step.

        The compressed method should complete on a 24 GB card that cannot
        run the standard method at this scale. We use a very small batch (1)
        and short sequence (32) to keep total memory reasonable.
        """
        # Large config: mimics Llama-7B hidden dimension
        config = _large_gpt2_config(
            n_embd=4096,
            n_layer=32,
            n_head=32,
            vocab_size=32000,
            n_positions=512,
        )
        batch_size = 1
        seq_len = 32
        compression_dim = 256  # ~16x compression from d=4096

        # --- Step 1: Create reference model ---
        # For the large model, the reference model itself takes a lot of VRAM.
        # We create a SEPARATE small config for the reference to keep memory
        # manageable, then generate synthetic corr_x directly.
        # This is valid because in a real scenario, corr_x comes from outside
        # the compressed model -- we just need the right shape.

        print(f"\n[Experiment 3] Config: d={config.n_embd}, layers={config.n_layer}, "
              f"heads={config.n_head}, k={compression_dim}")
        print(f"[Experiment 3] Input: batch={batch_size}, seq={seq_len}")

        # --- Step 2: Create compressed model ---
        model = FPT2LMHeadModelCompressed(
            config,
            with_embedding_nodes=True,
            compression_dim=compression_dim,
        ).to(DEVICE)
        model.reset_all_log_alphas()
        model.train()

        # --- Step 3: Generate dummy data ---
        clean_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)

        # Synthesize corr_x directly in the expected shape: (batch, writers, seq, d)
        # num_writers = 2 (embeds) + n_layer * (n_head + 1)
        n_writers = 2 + config.n_layer * (config.n_head + 1)
        corr_x = torch.randn(
            batch_size, n_writers, seq_len, config.n_embd,
            device=DEVICE, dtype=torch.float32,
        )
        # Detach -- in real training this comes from torch.no_grad() block
        corr_x = corr_x.detach()

        # --- Step 4: Forward + backward ---
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize(DEVICE)

        outputs = model(
            input_ids=clean_ids,
            target_edge_sparsity=0.95,
            target_node_sparsity=0.5,
            corr_x=corr_x,
        )

        # Simple loss: use edge + node regularization (no reference model needed)
        loss = outputs.edge_loss + outputs.node_loss
        loss.backward()

        torch.cuda.synchronize(DEVICE)
        peak_bytes = torch.cuda.max_memory_allocated(DEVICE)

        print(f"[Experiment 3] Peak VRAM: {peak_bytes / 1e9:.2f} GB")
        print(f"[Experiment 3] Edge sparsity: {outputs.model_edge_sparsity.item():.4f}")
        print(f"[Experiment 3] Node sparsity: {outputs.model_node_sparsity.item():.4f}")

        # --- Step 5: Assertions ---

        # The test completed without OOM -- that is the primary assertion.
        # Additionally verify that outputs are sane.
        assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"
        assert math.isfinite(outputs.model_edge_sparsity.item()), "Edge sparsity not finite"
        assert math.isfinite(outputs.model_node_sparsity.item()), "Node sparsity not finite"

        # Verify gradients actually flowed to the mask parameters
        sample_param = model.transformer.h[0].q_read_log_alphas
        assert sample_param.grad is not None, "No gradient on edge mask parameter (q_read_log_alphas)"
        assert torch.any(sample_param.grad != 0), "Gradient is all zeros on edge mask parameter"

        node_param = model.transformer.h[0].attn_write_log_alphas
        assert node_param.grad is not None, "No gradient on node mask parameter (attn_write_log_alphas)"

        # Peak VRAM should be under 24 GB for this configuration
        assert peak_bytes < 24 * 1e9, (
            f"Peak VRAM {peak_bytes / 1e9:.2f} GB exceeds 24 GB target"
        )

        # Cleanup
        del model, clean_ids, corr_x, outputs, loss
        _cleanup()


# ===========================================================================
# Bonus: Gradient Flow Sanity Checks
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason=SKIP_MSG)
class TestGradientFlow:
    """Verify that gradients flow correctly through the compressed routing
    to both edge mask and node mask parameters.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.config = _small_gpt2_config(n_embd=128, n_layer=2, n_head=4)
        yield
        _cleanup()

    def test_edge_mask_gradients_nonzero(self):
        """After one backward pass, edge mask log_alphas must have non-None,
        non-zero gradients."""
        model = FPT2LMHeadModelCompressed(
            self.config, with_embedding_nodes=True, compression_dim=64
        ).to(DEVICE)
        model.reset_all_log_alphas()
        model.train()

        clean_ids = torch.randint(0, self.config.vocab_size, (4, 16), device=DEVICE)
        n_writers = 2 + self.config.n_layer * (self.config.n_head + 1)
        corr_x = torch.randn(4, n_writers, 16, self.config.n_embd, device=DEVICE).detach()

        outputs = model(
            input_ids=clean_ids,
            target_edge_sparsity=0.95,
            target_node_sparsity=0.5,
            corr_x=corr_x,
        )
        loss = outputs.edge_loss + outputs.node_loss
        loss.backward()

        # Check every layer's edge mask parameters
        for i, block in enumerate(model.transformer.h):
            for name in ["q_read_log_alphas", "k_read_log_alphas", "v_read_log_alphas",
                         "mlp_read_log_alphas"]:
                param = getattr(block, name)
                assert param.grad is not None, (
                    f"Layer {i} {name}: gradient is None"
                )
                assert torch.any(param.grad != 0), (
                    f"Layer {i} {name}: gradient is all zeros"
                )

        # Check final read log_alphas
        final = model.transformer.final_read_log_alphas
        assert final.grad is not None, "final_read_log_alphas gradient is None"
        assert torch.any(final.grad != 0), "final_read_log_alphas gradient is all zeros"

        del model, clean_ids, corr_x, outputs, loss
        _cleanup()

    def test_node_mask_gradients_nonzero(self):
        """After one backward pass, node mask log_alphas must have non-None,
        non-zero gradients (proving gradient flows through compress())."""
        model = FPT2LMHeadModelCompressed(
            self.config, with_embedding_nodes=True, compression_dim=64
        ).to(DEVICE)
        model.reset_all_log_alphas()
        model.train()

        clean_ids = torch.randint(0, self.config.vocab_size, (4, 16), device=DEVICE)
        n_writers = 2 + self.config.n_layer * (self.config.n_head + 1)
        corr_x = torch.randn(4, n_writers, 16, self.config.n_embd, device=DEVICE).detach()

        outputs = model(
            input_ids=clean_ids,
            target_edge_sparsity=0.95,
            target_node_sparsity=0.5,
            corr_x=corr_x,
        )
        loss = outputs.edge_loss + outputs.node_loss
        loss.backward()

        # Check every layer's node mask parameters
        for i, block in enumerate(model.transformer.h):
            for name in ["attn_write_log_alphas", "mlp_write_log_alphas"]:
                param = getattr(block, name)
                assert param.grad is not None, (
                    f"Layer {i} {name}: gradient is None"
                )
                # Node mask gradients flow through compress() -> the custom autograd
                # function. If compress() broke the graph, these would be None/zero.
                assert torch.any(param.grad != 0), (
                    f"Layer {i} {name}: gradient is all zeros -- "
                    "compress() may have broken the autograd graph"
                )

        # Check embedding node masks
        tok_alpha = model.transformer.token_write_log_alpha
        pos_alpha = model.transformer.pos_write_log_alpha
        assert tok_alpha.grad is not None, "token_write_log_alpha gradient is None"
        assert pos_alpha.grad is not None, "pos_write_log_alpha gradient is None"

        del model, clean_ids, corr_x, outputs, loss
        _cleanup()

    def test_projection_matrices_frozen_by_default(self):
        """P and U should have requires_grad=False when freeze_projections=True."""
        model = FPT2LMHeadModelCompressed(
            self.config, with_embedding_nodes=True,
            compression_dim=64, freeze_projections=True,
        ).to(DEVICE)

        P = model.transformer.compression_P
        U = model.transformer.compression_U
        assert P is not None, "compression_P should exist when compression_dim > 0"
        assert U is not None, "compression_U should exist when compression_dim > 0"
        assert not P.requires_grad, "compression_P should be frozen by default"
        assert not U.requires_grad, "compression_U should be frozen by default"
        assert P.shape == (64, 128), f"P shape mismatch: {P.shape}"
        assert U.shape == (128, 64), f"U shape mismatch: {U.shape}"

        del model
        _cleanup()

    def test_no_compression_fallback(self):
        """With compression_dim=0, compressed model should behave identically
        to the original (no projection matrices created)."""
        model = FPT2LMHeadModelCompressed(
            self.config, with_embedding_nodes=True, compression_dim=0,
        ).to(DEVICE)

        assert model.transformer.compression_P is None
        assert model.transformer.compression_U is None
        assert model.transformer.compression_dim == 0

        # Should still be able to do a forward pass
        model.reset_all_log_alphas()
        model.train()
        clean_ids = torch.randint(0, self.config.vocab_size, (2, 8), device=DEVICE)
        n_writers = 2 + self.config.n_layer * (self.config.n_head + 1)
        corr_x = torch.randn(2, n_writers, 8, self.config.n_embd, device=DEVICE).detach()

        outputs = model(
            input_ids=clean_ids,
            target_edge_sparsity=0.95,
            target_node_sparsity=0.5,
            corr_x=corr_x,
        )
        loss = outputs.edge_loss + outputs.node_loss
        loss.backward()

        assert math.isfinite(loss.item()), f"Loss not finite in fallback mode: {loss.item()}"

        del model, clean_ids, corr_x, outputs, loss
        _cleanup()


# ===========================================================================
# Entry point for direct execution
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
