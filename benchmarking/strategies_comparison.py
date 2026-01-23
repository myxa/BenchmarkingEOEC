"""
Strategies Comparison Module

=====================

Computes mean correlation coefficients between all pairs of subjects
across 6 denoising strategies (+GSR variants).

Core function:
strategies_comparison() - Compute 32×32 correlation matrix between pipelines

Approach:
- Loads FC matrices for each strategy (1-6) + AROMA (aggr/nonaggr) × condition (close/open) × GSR
- Vectorizes upper triangle of FC matrices
- Computes mean Pearson r between each pair of pipelines across subjects

Usage:
python -m benchmarking.strategies_comparison --config configs/strategies_comparison.yaml

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Optional, Union

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from nilearn.connectome import sym_matrix_to_vec
import yaml
from tqdm import tqdm

from data_utils.paths import resolve_data_root
from data_utils.fc import ConnectomeTransformer
from data_utils.hcpex import preprocess_hcpex_timeseries


# =============================================================================
# FC loading utilities (adapted from qc_fc.py)
# =============================================================================

def _build_strategy_string(strategy: Union[int, str]) -> str:
    """
    Build strategy string for filename.
    
    Handles both numeric strategies (1-6) and AROMA strategies.
    """
    if isinstance(strategy, int):
        return str(strategy)
    return str(strategy)


def load_fc_for_strategies_comparison(
    site: Literal["china", "ihb"],
    condition: Literal["close", "open"],
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    fc_type: str,
    session_idx: int = 0,
    coverage_mask: Optional[np.ndarray] = None,
    data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load or compute FC for strategies comparison (China dataset).
    
    For glasso: loads precomputed FC.
    For corr/partial/tangent: loads timeseries and computes FC.
    
    Parameters
    ----------
    site : 'china' or 'ihb'
    condition : 'close' or 'open'
    atlas : str
    strategy : int or str (1-6 or 'AROMA_aggr', 'AROMA_nonaggr')
    gsr : 'GSR' or 'noGSR'
    fc_type : 'corr', 'partial', 'tangent', or 'glasso'
    session_idx : int ('close' = 1, 'open' = 0)
    coverage_mask : np.ndarray, optional (good ROIs mask)
    data_path : str, optional
    
    Returns
    -------
    fc_matrices : np.ndarray shape (n_subjects, n_rois, n_rois)
        FC matrices for all subjects
    """
    data_root = resolve_data_root(data_path)
    strategy_str = _build_strategy_string(strategy)
    
    if fc_type == 'glasso':
        # Load precomputed glasso
        glasso_dir = data_root / "glasso_precomputed_fc" / site / atlas
        filename = f"{site}_{condition}_{atlas}_strategy-{strategy_str}_{gsr}_glasso.npy"
        filepath = glasso_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Precomputed glasso not found: {filepath}")
        
        fc_matrices = np.load(filepath)

        if condition == "close" and fc_matrices.ndim == 4:
            fc_matrices = fc_matrices[:, :, :, session_idx]

        return fc_matrices
    
    else:
        # Load timeseries and compute FC
        ts_dir = data_root / f"timeseries_{site}" / atlas
        filename = f"{site}_{condition}_{atlas}_strategy-{strategy_str}_{gsr}.npy"
        filepath = ts_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Timeseries not found: {filepath}")
        
        ts = np.load(filepath)
        if condition == "close" and ts.ndim == 4:
            ts = ts[:, :, :, session_idx]
        
        # Apply coverage mask
        if coverage_mask is not None and atlas != "HCPex":
            ts = ts[:, :, coverage_mask]

        if atlas == "HCPex":
            hcpex_mask_pth = data_root / 'coverage' / f"hcp_mask.npy"
            hcpex_coverage_pth = data_root / 'coverage'
            ts = preprocess_hcpex_timeseries(ts, site=site, 
                                             mask_path=hcpex_mask_pth,
                                             coverage_dir=hcpex_coverage_pth)

        
        transformer = ConnectomeTransformer(
            kind=fc_type, 
            vectorize=False,  # Return full matrices
        )
        return transformer.fit_transform(ts)


# =============================================================================
# Main strategies comparison computation
# =============================================================================

def compute_strategies_comparison(
    site: Literal["china", "ihb"],
    closed_path: str,
    open_path: str,
    fc_type: str,
    atlas: str,
    coverage_mask: Optional[np.ndarray] = None,
    data_path: Optional[str] = None,
    include_aroma: bool = True,
    aroma_strategies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute strategies comparison matrix for China dataset including AROMA.

    Loads FC for:
    - 6 numeric strategies (1-6) × close/open × GSR/noGSR = 24 pipelines
    - (optional) AROMA strategies × close/open × GSR/noGSR

    Parameters
    ----------
    site : 'china' or 'ihb'
    closed_path, open_path : str
        Kept for API compatibility (not used directly if loader uses data_path),
        but can be forwarded if needed.
    fc_type : str
        'corr', 'partial', 'tangent', or 'glasso'
    atlas : str
        Atlas name
    coverage_mask : np.ndarray, optional
        ROI mask
    data_path : str, optional
        Root data path
    include_aroma : bool
        Whether to include AROMA strategies
    aroma_strategies : list[str], optional
        Defaults to ['AROMA_aggr', 'AROMA_nonaggr']

    Returns
    -------
    pd.DataFrame
        Square DataFrame matrix with mean subject-wise correlations between all pipelines.
    """
    numeric_strategies: List[tuple[Union[int, str], str]] = [
        (1, "24P"),
        (2, "aCompCor+12P"),
        (3, "aCompCor50+12P"),
        (4, "aCompCor+24P"),
        (5, "aCompCor50+24P"),
        (6, "atCompCor50+24P"),
    ]

    if aroma_strategies is None:
        aroma_strategies = ["AROMA_aggr", "AROMA_nonaggr"]

    all_strategies: List[tuple[Union[int, str], str]] = list(numeric_strategies)
    if include_aroma:
        all_strategies.extend([(s, s) for s in aroma_strategies])

    data: Dict[str, np.ndarray] = {}

    for strategy, label in all_strategies:
        for gsr in ("noGSR", "GSR"):
            # close
            cl = load_fc_for_strategies_comparison(
                site=site,
                condition="close",
                atlas=atlas,
                strategy=strategy,
                gsr=gsr,
                fc_type=fc_type,
                coverage_mask=coverage_mask,
                data_path=data_path,
            )
            data[f"close_{label}" + ("_GSR" if gsr == "GSR" else "")] = sym_matrix_to_vec(cl)

            # open
            op = load_fc_for_strategies_comparison(
                site=site,
                condition="open",
                atlas=atlas,
                strategy=strategy,
                gsr=gsr,
                fc_type=fc_type,
                coverage_mask=coverage_mask,
                data_path=data_path,
            )
            data[f"open_{label}" + ("_GSR" if gsr == "GSR" else "")] = sym_matrix_to_vec(op)

    keys = sorted(data.keys())
    n_pipelines = len(keys)
    n_subjects = next(iter(data.values())).shape[0]

    out = np.zeros((n_pipelines, n_pipelines), dtype=float)

    for i, key_i in enumerate(keys):
        vec_i = data[key_i]
        for j, key_j in enumerate(keys):
            vec_j = data[key_j]
            corrs = [
                np.corrcoef(vec_i[sub], vec_j[sub])[0, 1]
                for sub in range(n_subjects)
            ]
            out[i, j] = float(np.mean(corrs))

    return pd.DataFrame(data=out, columns=keys, index=keys)


def load_hcpex_mask(
    data_path: Optional[str] = None,
) -> np.ndarray:
    """Load HCPex mask."""
    data_root = resolve_data_root(data_path)
    mask = data_root / "coverage" / f"hcp_mask.npy"
    
    if not mask.exists():
        raise FileNotFoundError(f"Coverage file not found: {mask}")
    
    mask = np.load(mask).astype(float)
    #np.loadtxtchina_skipped_rois_HCPex.txt

    return mask

def load_coverage_mask(
    atlas: str,
    data_path: Optional[str] = None,
    threshold: float = 0.1,
) -> np.ndarray:
    """Load IHB coverage mask."""
    data_root = resolve_data_root(data_path)
    coverage_file = data_root / "coverage" / f"ihb_{atlas}_parcel_coverage.npy"
    
    if not coverage_file.exists():
        raise FileNotFoundError(f"Coverage file not found: {coverage_file}")
    
    coverage = np.load(coverage_file).astype(float)
    return coverage >= threshold


# =============================================================================
# Config-driven execution
# =============================================================================
def run_strategies_comparison_from_config(config: dict) -> pd.DataFrame:
    """
    Run strategies comparison and save one matrix per atlas (fc_type fixed).

    Config keys
    ----------
    - site: 'china' or 'ihb'
    - data_path: str
    - atlases: list[str]
    - fc_type: str                      # NOTE: single value
    - use_coverage_mask: bool
    - coverage_threshold: float
    - output_dir: str                   # directory for per-atlas CSVs

    Returns
    -------
    pd.DataFrame
        Small summary table with saved files (one row per atlas).
    """
    site = config.get("site", "ihb")
    data_path = Path(config.get("data_path")).expanduser().resolve()
    atlases = config.get("atlases", ["AAL", "Schaefer200", "Brainnetome", "HCPex"])

    fc_type = config.get("fc_type", None)
    if fc_type is None:
        raise ValueError("Config must contain 'fc_type' (single value), e.g. corr, partial, tangent, glasso")

    use_coverage_mask = config.get("use_coverage_mask", True)
    coverage_threshold = config.get("coverage_threshold", 0.1)

    output_dir = Path(config.get("output_dir", "strategies_comparison")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load coverage masks
    coverage_masks = {}
    if use_coverage_mask:
        for atlas in atlases:
            try:
                coverage_masks[atlas] = load_coverage_mask(atlas, data_path, coverage_threshold)
            except FileNotFoundError:
                print(f"Warning: Coverage mask not found for {atlas}, skipping masking")
                coverage_masks[atlas] = None

    print(f"Running strategies comparison for {len(atlases)} atlases (fc_type={fc_type})")

    summary_rows = []

    for atlas in tqdm(atlases, desc="Atlases"):
        coverage_mask = coverage_masks.get(atlas) if use_coverage_mask else None

        try:
            df_mat = compute_strategies_comparison(
                site=site,
                closed_path=data_path,
                open_path=data_path,
                fc_type=fc_type,
                atlas=atlas,
                coverage_mask=coverage_mask,
                data_path=data_path,
            )

            out_file = output_dir / site / atlas / f"{site}_{atlas}_{fc_type}_strategies_comparison.csv"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df_mat.to_csv(out_file, index=True)

            summary_rows.append({
                "site": site,
                "atlas": atlas,
                "fc_type": fc_type,
                "output_file": str(out_file),
                "n_pipelines": df_mat.shape[0],
            })

        except Exception as e:
            print(f"Error for {atlas}/{fc_type}: {e}")
            summary_rows.append({
                "site": site,
                "atlas": atlas,
                "fc_type": fc_type,
                "output_file": None,
                "n_pipelines": None,
                "error": str(e),
            })

    return pd.DataFrame(summary_rows)



def main():
    parser = argparse.ArgumentParser(
        description='Compute strategies comparison (mean corr between pipelines)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file',
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override output path from config',
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Override output if specified
    if args.output:
        config['output'] = args.output
    
    print("=" * 60)
    print("Strategies Comparison Analysis")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Atlases: {config.get('atlases', ['all'])}")
    print(f"FC types: {config.get('fc_type', '')}")
    
    # Run analysis
    df = run_strategies_comparison_from_config(config)
    
    # Save results
    
    print("\nDone!")


if __name__ == '__main__':
    main()
