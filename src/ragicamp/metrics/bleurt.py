"""BLEURT metric implementation with lazy loading and proper GPU cleanup."""

import gc
import os
from pathlib import Path
from typing import Any, Optional

# Fix matplotlib backend BEFORE any imports that might use it
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

# Configure TensorFlow to use memory growth (don't allocate all GPU memory)
# This MUST be done before importing TensorFlow
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from ragicamp.metrics.base import Metric

# Available BLEURT checkpoints (small ones for faster download/inference)
BLEURT_CHECKPOINTS = {
    "BLEURT-20": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip",
    "BLEURT-20-D12": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip",
    "BLEURT-20-D6": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip",
    "BLEURT-20-D3": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip",  # Smallest/fastest
}


class BLEURTMetric(Metric):
    """BLEURT (Bilingual Evaluation Understudy with Representations from Transformers).

    A learned metric for natural language generation that correlates well with
    human judgments.

    Uses lazy loading - model is only loaded when compute() is called,
    and unloaded immediately after to free GPU memory.

    Note: BLEURT-20-D3 is recommended for speed (smallest model).
    """

    def __init__(self, checkpoint: str = "BLEURT-20-D3", **kwargs: Any):
        """Initialize BLEURT metric.

        Args:
            checkpoint: BLEURT checkpoint to use
                Options: BLEURT-20, BLEURT-20-D12, BLEURT-20-D6, BLEURT-20-D3
                Default: BLEURT-20-D3 (smallest/fastest)
            **kwargs: Additional configuration
        """
        super().__init__(name="bleurt", **kwargs)
        self.checkpoint = checkpoint
        self._scorer: Optional[Any] = None  # Lazy loaded
        self._checkpoint_path: Optional[str] = None  # Cache resolved path

    def _load_scorer(self) -> None:
        """Load the BLEURT model (lazy loading)."""
        if self._scorer is not None:
            return

        try:
            from bleurt import score as bleurt_score
        except ImportError as e:
            raise ImportError(
                "BLEURT is required for BLEURTMetric. "
                "Install with: uv sync (already included in dependencies)"
            ) from e

        print(f"  📥 Loading BLEURT model: {self.checkpoint}")

        # Check cache first before trying anything else
        cache_path = Path.home() / ".cache" / "bleurt" / self.checkpoint
        if cache_path.exists():
            self._checkpoint_path = str(cache_path)
            self._scorer = bleurt_score.BleurtScorer(self._checkpoint_path)
            print("  ✓ Loaded from cache")
            return

        # Try to load checkpoint by name (might work if installed via pip)
        try:
            self._scorer = bleurt_score.BleurtScorer(self.checkpoint)
            return
        except Exception:
            pass

        # Download checkpoint
        print(f"  BLEURT checkpoint '{self.checkpoint}' not found locally.")
        print("  Downloading (this only happens once)...")

        try:
            self._checkpoint_path = self._download_checkpoint(self.checkpoint)
            self._scorer = bleurt_score.BleurtScorer(self._checkpoint_path)
            print("  ✓ BLEURT checkpoint loaded successfully")
        except Exception as download_error:
            raise RuntimeError(
                f"Failed to load/download BLEURT checkpoint '{self.checkpoint}'.\n"
                f"Error: {download_error}\n\n"
                f"Available checkpoints: {', '.join(BLEURT_CHECKPOINTS.keys())}\n"
                f"Try using a smaller checkpoint: BLEURT-20-D3 (fastest)\n\n"
                f"Manual download:\n"
                f"  mkdir -p ~/.cache/bleurt\n"
                f"  cd ~/.cache/bleurt\n"
                f"  wget {BLEURT_CHECKPOINTS.get(self.checkpoint, 'URL_NOT_FOUND')}\n"
                f"  unzip {self.checkpoint}.zip"
            ) from download_error

    def _unload_scorer(self) -> None:
        """Unload the BLEURT model to free GPU memory."""
        if self._scorer is None:
            return

        # Delete the scorer
        del self._scorer
        self._scorer = None

        # Clear TensorFlow session/memory
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
            # Reset GPU memory
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.reset_memory_stats(gpu)
        except Exception:
            pass

        # Also clear PyTorch if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        print("  🗑️  BLEURT model unloaded")

    def _download_checkpoint(self, checkpoint: str) -> str:
        """Download BLEURT checkpoint.

        Args:
            checkpoint: Checkpoint name

        Returns:
            Path to downloaded checkpoint
        """
        import urllib.request
        import zipfile

        if checkpoint not in BLEURT_CHECKPOINTS:
            raise ValueError(
                f"Unknown checkpoint: {checkpoint}\n"
                f"Available: {', '.join(BLEURT_CHECKPOINTS.keys())}"
            )

        url = BLEURT_CHECKPOINTS[checkpoint]
        cache_dir = Path.home() / ".cache" / "bleurt"
        cache_dir.mkdir(parents=True, exist_ok=True)

        zip_path = cache_dir / f"{checkpoint}.zip"
        extract_path = cache_dir / checkpoint

        # Download if not already cached
        if not extract_path.exists():
            print(f"  Downloading {checkpoint} from {url}...")
            urllib.request.urlretrieve(url, zip_path)

            print(f"  Extracting {checkpoint}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_dir)

            # Clean up zip file
            zip_path.unlink()
            print(f"  ✓ Downloaded and extracted to {extract_path}")

        return str(extract_path)

    def compute(
        self, predictions: list[str], references: list[str], **kwargs: Any
    ) -> dict[str, float]:
        """Compute BLEURT scores with automatic batching and OOM retry.

        Loads model, computes scores in batches, then unloads to free GPU memory.
        If OOM occurs, automatically retries with smaller batch size.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)

        Returns:
            Dict with average BLEURT score (for overall metrics)
        """
        try:
            # Load model (lazy)
            self._load_scorer()

            # Start with batch_size=64, halve on OOM
            batch_size = 64
            min_batch_size = 1

            while batch_size >= min_batch_size:
                try:
                    all_scores = []

                    for i in range(0, len(predictions), batch_size):
                        batch_preds = predictions[i : i + batch_size]
                        batch_refs = references[i : i + batch_size]
                        batch_scores = self._scorer.score(references=batch_refs, candidates=batch_preds)
                        all_scores.extend(batch_scores)

                    # Store per-item scores for detailed analysis
                    self._last_scores = list(all_scores)

                    # Only return the mean score (not individual scores)
                    return {"bleurt": float(sum(all_scores) / len(all_scores)) if all_scores else 0.0}

                except Exception as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "oom" in error_msg or "resource exhausted" in error_msg:
                        # Clear memory and retry with smaller batch
                        gc.collect()
                        try:
                            import tensorflow as tf
                            tf.keras.backend.clear_session()
                        except Exception:
                            pass

                        old_batch_size = batch_size
                        batch_size = batch_size // 2
                        if batch_size >= min_batch_size:
                            print(f"    ⚠ OOM with batch_size={old_batch_size}, retrying with {batch_size}")
                        else:
                            raise RuntimeError(
                                f"BLEURT OOM even with batch_size=1. "
                                f"Text too long or GPU memory insufficient."
                            ) from e
                    else:
                        raise

            # Should not reach here
            raise RuntimeError("BLEURT computation failed")

        finally:
            # ALWAYS unload after computation to free GPU
            self._unload_scorer()

    def get_per_item_scores(self) -> list[float]:
        """Get per-item scores from last compute() call."""
        return getattr(self, "_last_scores", [])
