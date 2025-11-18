"""BLEURT metric implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

# Fix matplotlib backend BEFORE any imports that might use it
# This prevents errors when running in non-interactive environments (scripts vs notebooks)
if 'MPLBACKEND' in os.environ:
    # Change to non-interactive backend for scripts
    os.environ['MPLBACKEND'] = 'Agg'

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
        
        # Lazy import to avoid requiring bleurt unless used
        try:
            from bleurt import score as bleurt_score
        except ImportError:
            raise ImportError(
                "BLEURT is required for BLEURTMetric. "
                "Install with: uv sync (already included in dependencies)"
            )
        
        # Try to load checkpoint, download if needed
        try:
            self.scorer = bleurt_score.BleurtScorer(checkpoint)
        except Exception as e:
            # Try to download checkpoint
            print(f"BLEURT checkpoint '{checkpoint}' not found locally.")
            print("Attempting to download...")
            
            try:
                checkpoint_path = self._download_checkpoint(checkpoint)
                self.scorer = bleurt_score.BleurtScorer(checkpoint_path)
                print(f"✓ BLEURT checkpoint loaded successfully")
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to load/download BLEURT checkpoint '{checkpoint}'.\n"
                    f"Error: {download_error}\n\n"
                    f"Available checkpoints: {', '.join(BLEURT_CHECKPOINTS.keys())}\n"
                    f"Try using a smaller checkpoint: BLEURT-20-D3 (fastest)\n\n"
                    f"Manual download:\n"
                    f"  mkdir -p ~/.cache/bleurt\n"
                    f"  cd ~/.cache/bleurt\n"
                    f"  wget {BLEURT_CHECKPOINTS.get(checkpoint, 'URL_NOT_FOUND')}\n"
                    f"  unzip {checkpoint}.zip"
                )
    
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
            print(f"Downloading {checkpoint} from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            
            print(f"Extracting {checkpoint}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            
            # Clean up zip file
            zip_path.unlink()
            print(f"✓ Downloaded and extracted to {extract_path}")
        
        return str(extract_path)
    
    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        **kwargs: Any
    ) -> Dict[str, float]:
        """Compute BLEURT scores.
        
        Returns:
            Dict with average BLEURT score (for overall metrics)
        """
        # Handle multiple references - take first one for now
        refs = []
        for ref in references:
            if isinstance(ref, list):
                refs.append(ref[0] if ref else "")
            else:
                refs.append(ref)
        
        # Compute BLEURT scores
        scores = self.scorer.score(references=refs, candidates=predictions)
        
        # Only return the mean score (not individual scores)
        # Individual scores are handled by compute_single() for per-question metrics
        return {
            "bleurt": float(sum(scores) / len(scores)) if scores else 0.0
        }

