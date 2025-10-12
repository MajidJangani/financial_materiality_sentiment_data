"""
ESG MATERIALITY ANALYSIS ENGINE
================================
Core analytical functions for ESG driver materiality assessment.

This module implements the complete two-framework pipeline:
- Framework 1: Feature Selection (27 → 12 → 6 drivers)
- Framework 2: Temporal Validation (26-week windows with regime detection)
- Focusing on 26-week Framework 2 with dual-level regime detection

Author: Majid Jangani
Book: ESG Financial Materiality Assessment
Version: 2.0 (Book Publication Edition)
date: 2025-10-08
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.stats import spearmanr
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# =================================================================
# CONFIGURATION
# =================================================================

DEFAULT_CONFIG = {
    'target_creation': {
        'min_class_size': 15,
        'base_multiplier': 0.75,
        'min_class_proportion': 0.15,
        'balance_quality_threshold': 0.60,
        'signal_preservation_threshold': 0.3,
        'fallback_percentiles': {
            '3class': [33, 67],
            '5class': [20, 40, 60, 80]
        },
        'skew_adjustment_factor': 0.03,
        'use_dynamic_multiplier': True,
    },
    'framework1': {
        'use_binary_enhancement': True,
        'target_features_stage1': 12,  # Stage 1: 27 → 12
        'target_features_stage2': 6,   # Stage 2: 12 → 6 
        'mi_weight': 0.5,
        'binary_auc_weight': 0.3,
        'rf_weight': 0.2,
        'cv_splits': 3,
    },
    'framework2': {
        'window_size': 39,            # approx 9 months
        'step_size': 13,              # 13-week steps (quarterly)
        'min_window_samples': 52,
        'enable_individual_regime_tracking': True,
        'enable_pillar_regime_detection': True,
        'regime_min_length': 3,       # Minimum 3 windows for regime
    }
}

# =================================================================
# SECTION 1: TARGET CREATION (CHAPTER 4)
# =================================================================

def create_three_class_targets_skewness_aware(returns: np.ndarray, 
                                            verbose: bool = True, 
                                            config: Optional[Dict] = None) -> Tuple[List[int], Dict, Dict[str, bool]]:
    """
    Create sophisticated 3-class performance targets with skewness handling.
    
    This function creates three performance categories for ESG materiality analysis:
    - Class 0: ESG Lagging (bottom performers)
    - Class 1: ESG Stable (middle performers)  
    - Class 2: ESG Leading (top performers)
    
    Uses Median Absolute Deviation (MAD) for robust statistical boundaries
    that remain stable across different market regimes (bull/bear markets).
    
    Args:
        returns: Weekly log returns, shape (n_weeks,)
        verbose: Whether to print diagnostic information
        config: Configuration dict for algorithm parameters
        
    Returns:
        targets: List of class labels {0, 1, 2}
        metadata: Dict with statistics and quality metrics
        quality_gates: Dict with validation flags
        
    Example:
        >>> returns = np.log(df['close_price'] / df['close_price'].shift(1)).dropna()
        >>> targets, metadata, quality_gates = create_three_class_targets_skewness_aware(returns)
        >>> print(f"Class distribution: {Counter(targets)}")
        Class distribution: Counter({0: 95, 1: 68, 2: 110})
    """
    
    if config is None:
        config = DEFAULT_CONFIG['target_creation']
    
    # Convert to numpy array and remove NaNs
    returns_array = np.array(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    # Validate sufficient data
    if len(returns_clean) < 50:
        raise ValueError(f"Insufficient data: {len(returns_clean)} samples (minimum 50 required)")
    
    # Calculate robust statistics
    median_ret = np.median(returns_clean)
    mad = np.median(np.abs(returns_clean - median_ret))
    
    # Calculate robust skewness
    if mad > 0:
        skewness = ((returns_clean - median_ret) ** 3).mean() / (mad ** 3)
    else:
        skewness = stats.skew(returns_clean)
    
    if verbose:
        print(f"  3-CLASS TARGET CREATION:")
        print(f"    Samples: {len(returns_clean)}")
        print(f"    Median Return: {median_ret:.4f}")
        print(f"    MAD: {mad:.4f}")
        print(f"    Robust Skewness: {skewness:.3f}")
    
    # Dynamic multiplier adjustment based on skewness
    if config.get('use_dynamic_multiplier', True):
        if abs(skewness) > 2.5:
            base_multiplier = 0.45
        elif abs(skewness) > 2.0:
            base_multiplier = 0.5
        elif abs(skewness) > 1.5:
            base_multiplier = 0.55
        elif abs(skewness) > 1.0:
            base_multiplier = 0.6
        elif abs(skewness) > 0.5:
            base_multiplier = 0.7
        else:
            base_multiplier = config.get('base_multiplier', 0.75)
        
        if verbose:
            print(f"    Base multiplier (skew-adjusted): {base_multiplier}")
    else:
        base_multiplier = config.get('base_multiplier', 0.75)
    
    # Fine-tuning for high skewness
    skew_adjustment_factor = config.get('skew_adjustment_factor', 0.025)
    
    if mad > 0 and abs(skewness) > 1.5:
        skew_adjustment = min(0.08, abs(skewness) * skew_adjustment_factor)
        
        if skewness < 0:  # Left-skewed
            multiplier_lower = base_multiplier + skew_adjustment
            multiplier_upper = base_multiplier - skew_adjustment
            if verbose:
                print(f"    Left-skew adjustment: +{skew_adjustment:.3f} lower, -{skew_adjustment:.3f} upper")
        else:  # Right-skewed
            multiplier_lower = base_multiplier - skew_adjustment
            multiplier_upper = base_multiplier + skew_adjustment
            if verbose:
                print(f"    Right-skew adjustment: -{skew_adjustment:.3f} lower, +{skew_adjustment:.3f} upper")
        
        threshold_lower = median_ret - (multiplier_lower * mad)
        threshold_upper = median_ret + (multiplier_upper * mad)
        multiplier = base_multiplier
    else:
        multiplier = base_multiplier
        threshold_lower = median_ret - (multiplier * mad)
        threshold_upper = median_ret + (multiplier * mad)
    
    # Fallback to percentiles if MAD is zero
    if mad == 0:
        threshold_lower = np.percentile(returns_clean, config['fallback_percentiles']['3class'][0])
        threshold_upper = np.percentile(returns_clean, config['fallback_percentiles']['3class'][1])
        if verbose:
            print(f"    MAD=0, using percentile fallback")
        method_used = 'three_class_percentile_fallback'
    else:
        method_used = 'three_class_skewness_aware_mad'
    
    if verbose:
        print(f"    Final thresholds: Lower={threshold_lower:.4f}, Upper={threshold_upper:.4f}")
    
    # Assign classes
    targets = []
    for ret in returns_clean:
        if ret < threshold_lower:
            targets.append(0)  # ESG Lagging
        elif ret < threshold_upper:
            targets.append(1)  # ESG Stable
        else:
            targets.append(2)  # ESG Leading
    
    # Validate class sizes
    counts = Counter(targets)
    min_class_size = config.get('min_class_size', 15)
    
    if any(counts.get(i, 0) < min_class_size for i in range(3)):
        if verbose:
            print(f"    Class size correction - using standard percentiles")
        threshold_lower = np.percentile(returns_clean, 33)
        threshold_upper = np.percentile(returns_clean, 67)
        
        targets = []
        for ret in returns_clean:
            if ret < threshold_lower:
                targets.append(0)
            elif ret < threshold_upper:
                targets.append(1)
            else:
                targets.append(2)
        
        counts = Counter(targets)
        method_used = 'three_class_percentile_corrected'
    
    # Calculate quality metrics
    total = len(targets)
    class_proportions = [counts.get(i, 0) / total for i in range(3)]
    
    target_proportion = 1.0 / 3
    balance_quality = 1 - (np.std(class_proportions) / target_proportion)
    signal_preservation = (counts.get(0, 0) + counts.get(2, 0)) / total
    
    # Quality gates
    quality_gates = {
        'sufficient_samples': total >= 50,
        'balance_acceptable': balance_quality >= config.get('balance_quality_threshold', 0.60),
        'signal_preserved': signal_preservation >= config.get('signal_preservation_threshold', 0.3),
        'all_classes_present': all(counts.get(i, 0) > 0 for i in range(3)),
        'skewness_manageable': abs(skewness) <= 3.0,
        'mad_reliable': mad > 1e-6
    }
    
    # Metadata
    metadata = {
        'class_distribution': dict(counts),
        'class_proportions': class_proportions,
        'balance_quality': balance_quality,
        'signal_preservation': signal_preservation,
        'thresholds': {'lower': threshold_lower, 'upper': threshold_upper},
        'statistics': {
            'median': median_ret,
            'mad': mad,
            'skewness': skewness,
            'method_used': method_used
        },
        'parameters': {
            'base_multiplier': base_multiplier,
            'skew_adjustment_applied': abs(skewness) > 1.5 and mad > 0,
            'dynamic_multiplier_used': config.get('use_dynamic_multiplier', True)
        },
        'quality_summary': {
            'overall_quality': 'excellent' if all(quality_gates.values()) else 
                              'good' if sum(quality_gates.values()) >= 5 else 'acceptable',
            'primary_concerns': [k for k, v in quality_gates.items() if not v]
        }
    }
    
    if verbose:
        print(f"    Class Distribution:")
        class_names = ['ESG Lagging', 'ESG Stable', 'ESG Leading']
        for i, name in enumerate(class_names):
            print(f"      {i}: {name:<15} {counts.get(i, 0):>4} ({class_proportions[i]*100:>5.1f}%)")
        print(f"    Balance Quality: {balance_quality:.3f}")
        print(f"    Signal Preservation: {signal_preservation:.3f}")
    
    return targets, metadata, quality_gates


def create_research_enhanced_binary_targets(returns: np.ndarray, 
                                          verbose: bool = True, 
                                          config: Optional[Dict] = None) -> Tuple[List[int], Dict]:
    """
    Create sophisticated binary performance targets for ESG value creation analysis.
    
    This function identifies companies demonstrating "Value Creation" (1) vs 
    "Baseline Performance" (0) using robust statistical boundaries.
    
    Purpose: Identify companies where ESG factors genuinely drive superior 
    financial performance, enabling focus on highest-impact ESG factors.
    
    Args:
        returns: Weekly log returns, shape (n_weeks,)
        verbose: Whether to print diagnostic information
        config: Configuration dict for algorithm parameters
        
    Returns:
        targets: Binary class labels {0, 1}
        metadata: Dict with statistics and quality metrics
        
    Example:
        >>> returns = np.log(df['close_price'] / df['close_price'].shift(1)).dropna()
        >>> targets, metadata = create_research_enhanced_binary_targets(returns)
        >>> print(f"Value Creation rate: {sum(targets)/len(targets):.1%}")
        Value Creation rate: 40.3%
    """
    
    if config is None:
        config = DEFAULT_CONFIG['target_creation']
    
    # Convert and clean
    returns_array = np.array(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 30:
        raise ValueError(f"Insufficient data: {len(returns_clean)} samples (minimum 30 required)")
    
    # Calculate robust statistics
    median_ret = np.median(returns_clean)
    raw_mad = np.median(np.abs(returns_clean - median_ret))
    
    # Robust skewness
    if raw_mad > 0:
        skewness = ((returns_clean - median_ret) ** 3).mean() / (raw_mad ** 3)
    else:
        skewness = stats.skew(returns_clean)
    
    if verbose:
        print(f"  BINARY TARGET CREATION:")
        print(f"    Samples: {len(returns_clean)}")
        print(f"    Median Return: {median_ret:.4f}")
        print(f"    MAD: {raw_mad:.4f}")
        print(f"    Robust Skewness: {skewness:.3f}")
    
    # Dynamic multiplier
    if abs(skewness) > 2.0:
        base_multiplier = 0.5
    elif abs(skewness) > 1.5:
        base_multiplier = 0.55
    elif abs(skewness) > 1.0:
        base_multiplier = 0.6
    else:
        base_multiplier = config.get('base_multiplier', 0.75)
    
    # Skewness adjustment
    skew_adjustment_factor = config.get('skew_adjustment_factor', 0.03)
    
    if raw_mad > 0 and abs(skewness) > 1.5:
        skew_adjustment = min(0.1, abs(skewness) * skew_adjustment_factor)
        
        if skewness < 0:  # Left-skewed
            threshold = median_ret + (base_multiplier - skew_adjustment) * raw_mad
            if verbose:
                print(f"    Left-skewed adjustment: -{skew_adjustment:.3f}")
        else:  # Right-skewed
            threshold = median_ret + (base_multiplier + skew_adjustment) * raw_mad
            if verbose:
                print(f"    Right-skewed adjustment: +{skew_adjustment:.3f}")
    else:
        threshold = median_ret + base_multiplier * raw_mad
    
    if verbose:
        print(f"    Final multiplier: {base_multiplier:.3f}")
        print(f"    Threshold (Value Creation): {threshold:.4f}")
    
    # Create binary classification
    targets = [1 if ret >= threshold else 0 for ret in returns_clean]
    counts = Counter(targets)
    total = len(targets)
    positive_rate = counts.get(1, 0) / total
    
    # Class size protection
    min_class_size = max(config.get('min_class_size', 15), int(total * 0.15))
    
    if counts.get(1, 0) < min_class_size or counts.get(0, 0) < min_class_size:
        if verbose:
            print(f"    Applying class size correction")
        fallback_threshold = np.percentile(returns_clean, 33)
        targets = [1 if ret >= fallback_threshold else 0 for ret in returns_clean]
        counts = Counter(targets)
        positive_rate = counts.get(1, 0) / total
        actual_threshold = fallback_threshold
    else:
        actual_threshold = threshold
    
    # Balance quality
    balance_quality = 1 - abs(positive_rate - 0.5) * 2
    
    if verbose:
        print(f"    ESG Baseline (0): {counts.get(0, 0)} ({(1-positive_rate)*100:.1f}%)")
        print(f"    ESG Value Creation (1): {counts.get(1, 0)} ({positive_rate*100:.1f}%)")
        print(f"    Balance Quality: {balance_quality:.3f}")
    
    metadata = {
        'threshold': actual_threshold,
        'method_used': 'research_enhanced_binary',
        'balance_quality': balance_quality,
        'positive_rate': positive_rate,
        'class_distribution': dict(counts),
        'skewness': skewness,
        'base_multiplier': base_multiplier,
        'median_return': median_ret,
        'mad': raw_mad
    }
    
    return targets, metadata


# =================================================================
# SECTION 2: FRAMEWORK 1 - (CHAPTER 5)
# =================================================================

def setup_conservative_cv(X, y, config=None):
    """
    Setup time-series aware cross-validation splits.
    
    Creates temporally-ordered validation splits that respect the fundamental
    principle: never use future information to predict the past.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        config: CV configuration dict
        
    Returns:
        List of (train_indices, val_indices) tuples
        
    Example:
        >>> cv_splits = setup_conservative_cv(X, y, {'n_splits': 3})
        >>> for fold, (train_idx, val_idx) in enumerate(cv_splits):
        >>>     print(f"Fold {fold}: Train {len(train_idx)}, Val {len(val_idx)}")
    """
    
    if config is None:
        config = {}
    
    n_splits = config.get('n_splits', 3)
    test_size = config.get('test_size', None)
    max_train_size = config.get('max_train_size', None)
    
    n_samples = len(X)
    
    # Adjust splits if test_size specified
    if test_size is not None:
        if isinstance(test_size, float):
            min_test_samples = int(n_samples * test_size)
        else:
            min_test_samples = test_size
        
        max_feasible_splits = max(1, (n_samples - min_test_samples) // min_test_samples)
        n_splits = min(n_splits, max_feasible_splits)
    
    # Handle None test_size
    if test_size is not None:
        if isinstance(test_size, float):
            test_size_param = int(n_samples * test_size)
        else:
            test_size_param = int(test_size)
    else:
        test_size_param = None
    
    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size_param,
        max_train_size=max_train_size
    )
    
    # Validate splits
    valid_splits = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        y_train_fold = np.array(y)[train_idx]
        y_val_fold = np.array(y)[val_idx]
        
        train_classes = len(np.unique(y_train_fold))
        val_classes = len(np.unique(y_val_fold))
        
        if train_classes >= 2 and val_classes >= 2:
            valid_splits.append((train_idx, val_idx))
        else:
            print(f"  Warning: Fold {fold} skipped (insufficient class diversity)")
    
    if not valid_splits:
        raise ValueError("No valid splits found. Consider reducing n_splits or adjusting data.")
    
    return valid_splits


def calculate_unified_importance_scores_with_binary(X, y_binary, y_3class, cv_splits, config=None):
    """
    Stage 1: Calculate MI and AUC importance scores (27 → 12 drivers).
    
    This is the first stage of feature selection using two complementary methods:
    1. Mutual Information: Captures any type of relationship (linear/non-linear)
    2. Binary AUC: Measures practical classification power
    
    Args:
        X: Feature matrix with all PBS drivers
        y_binary: Binary targets for AUC calculation
        y_3class: 3-class targets for MI calculation
        cv_splits: Cross-validation splits from setup_conservative_cv()
        config: Framework 1 configuration dict
        
    Returns:
        Dict with 'MI', 'AUC', and 'composite' importance scores
        
    Example:
        >>> importance_results = calculate_unified_importance_scores_with_binary(
        >>>     X, targets_binary, targets_3class, cv_splits
        >>> )
        >>> print(f"Top driver: {importance_results['composite'][0]}")
    """
    
    if config is None:
        config = DEFAULT_CONFIG['framework1']
    
    print(f"  STAGE 1: MI + AUC importance scoring (27 → 12)...")
    print(f"    Evaluating {X.shape[1]} ESG features across {len(cv_splits)} CV folds")
    
    # Storage for scores from each fold
    mi_scores_per_fold = []
    auc_scores_per_fold = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"    Processing CV fold {fold + 1}/{len(cv_splits)}...")
        
        X_fold = X.iloc[train_idx]
        y_binary_fold = np.array(y_binary)[train_idx]
        y_3class_fold = np.array(y_3class)[train_idx]
        
        # Mutual Information (3-class for richer information)
        mi_scores = mutual_info_classif(X_fold, y_3class_fold, random_state=42)
        mi_scores_per_fold.append(mi_scores)
        
        # Binary AUC
        auc_scores = []
        for i in range(X_fold.shape[1]):
            try:
                feature_values = X_fold.iloc[:, i].values
                auc = roc_auc_score(y_binary_fold, feature_values)
                auc = max(auc, 1 - auc)  # Direction-agnostic
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        
        auc_scores_per_fold.append(auc_scores)
    
    print(f"    Completed scoring across {len(cv_splits)} folds")
    
    # Aggregate results
    avg_mi = np.mean(mi_scores_per_fold, axis=0)
    std_mi = np.std(mi_scores_per_fold, axis=0)
    avg_auc = np.mean(auc_scores_per_fold, axis=0)
    std_auc = np.std(auc_scores_per_fold, axis=0)
    
    results = {
        'MI': list(zip(X.columns, avg_mi, std_mi)),
        'AUC': list(zip(X.columns, avg_auc, std_auc))
    }
    
    # Create composite scores
    print(f"    Creating composite scores...")
    composite_scores = []
    
    for i, feature in enumerate(X.columns):
        # Normalize MI
        mi_norm = avg_mi[i] / max(avg_mi) if max(avg_mi) > 0 else 0
        
        # Normalize AUC
        auc_norm = (avg_auc[i] - 0.5) * 2
        
        # Weighted combination
        total_weight = config['mi_weight'] + config['binary_auc_weight']
        mi_weight_adjusted = config['mi_weight'] / total_weight
        auc_weight_adjusted = config['binary_auc_weight'] / total_weight
        
        composite = mi_weight_adjusted * mi_norm + auc_weight_adjusted * auc_norm
        composite_scores.append((feature, composite, 0))
    
    composite_scores.sort(key=lambda x: x[1], reverse=True)
    results['composite'] = composite_scores
    
    print(f"    Top 5 drivers:")
    for i in range(min(5, len(composite_scores))):
        feature, score, _ = composite_scores[i]
        driver_num = int(feature.split('_')[1])
        pillar = get_esg_pillar(driver_num)
        print(f"      {i+1}. Driver {driver_num} ({pillar[:3]}): {score:.4f}")
    
    return results


def create_enhanced_composite_scores(importance_results: Dict, 
                                    config: Optional[Dict] = None) -> List[Dict]:
    """
    Create comprehensive driver profiles with multi-method importance scores.
    
    Transforms raw importance results into structured profiles with:
    - Composite scores from MI + AUC
    - ESG pillar classification
    - Binary classification power flags
    
    Args:
        importance_results: Output from calculate_unified_importance_scores_with_binary()
        config: Framework 1 configuration dict
        
    Returns:
        List of driver info dictionaries sorted by composite score
        
    Example:
        >>> composite_scores = create_enhanced_composite_scores(importance_results)
        >>> for driver in composite_scores[:3]:
        >>>     print(f"Driver {driver['driver_number']}: {driver['composite_score']:.3f}")
    """
    
    if config is None:
        config = DEFAULT_CONFIG['framework1']
    
    composite_scores = []
    
    for feature, score, _ in importance_results['composite']:
        driver_num = int(feature.split('_')[1])
        
        # Get individual scores
        mi_score = next(s for f, s, _ in importance_results['MI'] if f == feature)
        auc_score = next(s for f, s, _ in importance_results['AUC'] if f == feature)
        
        driver_info = {
            'feature': feature,
            'driver_number': driver_num,
            'pillar': get_esg_pillar(driver_num),
            'composite_score': score,
            'mi_score': mi_score,
            'auc_score': auc_score,
            'binary_classification_power': auc_score > 0.6
        }
        
        composite_scores.append(driver_info)
    
    return composite_scores


def select_balanced_drivers_enhanced(composite_scores, target_features=12):
    """
    Select top ESG drivers with pillar balancing (Stage 1: 27 → 12).
    
    Implements sophisticated selection that balances:
    - Statistical importance (composite scores)
    - ESG pillar representation (Environmental/Social/Governance)
    - Binary classification power
    
    Args:
        composite_scores: Output from create_enhanced_composite_scores()
        target_features: Number of drivers to select (default: 12)
        
    Returns:
        Tuple of (selected_drivers, pillar_counts)
        
    Example:
        >>> selected_12, pillar_dist = select_balanced_drivers_enhanced(
        >>>     composite_scores, target_features=12
        >>> )
        >>> print(f"Pillar distribution: {pillar_dist}")
        Pillar distribution: {'Environmental': 3, 'Social': 5, 'Governance': 4}
    """
    
    # Group by pillar
    pillar_drivers = {'Environmental': [], 'Social': [], 'Governance': []}
    
    for score_info in composite_scores:
        pillar = score_info['pillar']
        if pillar in pillar_drivers and score_info['driver_number'] > 0:
            pillar_drivers[pillar].append(score_info)
    
    # Sort within each pillar
    for pillar in pillar_drivers:
        pillar_drivers[pillar].sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Determine allocation
    if target_features <= 8:
        target_allocation = {'Environmental': 2, 'Social': 3, 'Governance': 3}
    elif target_features <= 12:
        target_allocation = {'Environmental': 3, 'Social': 5, 'Governance': 4}
    else:
        target_allocation = {'Environmental': 4, 'Social': 6, 'Governance': 5}
    
    # Adjust allocation
    total_allocated = sum(target_allocation.values())
    if total_allocated != target_features:
        diff = target_features - total_allocated
        target_allocation['Social'] += diff
    
    # First pass: select from each pillar
    selected_drivers = []
    pillar_counts = {'Environmental': 0, 'Social': 0, 'Governance': 0}
    
    for pillar, target_count in target_allocation.items():
        available = pillar_drivers[pillar]
        selected_count = min(target_count, len(available))
        
        for i in range(selected_count):
            selected_drivers.append(available[i])
            pillar_counts[pillar] += 1
    
    # Second pass: fill remaining slots
    remaining_slots = target_features - len(selected_drivers)
    if remaining_slots > 0:
        selected_features = {d['feature'] for d in selected_drivers}
        remaining_candidates = [s for s in composite_scores if s['feature'] not in selected_features]
        
        remaining_candidates.sort(
            key=lambda x: (x['binary_classification_power'], x['composite_score']), 
            reverse=True
        )
        
        for i in range(min(remaining_slots, len(remaining_candidates))):
            driver_info = remaining_candidates[i]
            selected_drivers.append(driver_info)
            pillar_counts[driver_info['pillar']] += 1
    
    print(f"  Selected {len(selected_drivers)} drivers with pillar distribution: {pillar_counts}")
    
    return selected_drivers, pillar_counts


def refine_features_with_rf(X_selected_12, y, selected_drivers_12, target_final=6, verbose=True):
    """
    Stage 2: Random Forest refinement (12 → 6 drivers).
    
    CHANGED FROM PRODUCTION VERSION: Now selects 6 final drivers (was 8).
    
    Uses Random Forest to identify the most important features from the 
    pre-selected 12 drivers. With improved feature-to-sample ratio (274/12 ≈ 23),
    Random Forest can confidently identify non-linear relationships and 
    feature interactions.
    
    Args:
        X_selected_12: DataFrame with 12 pre-selected features
        y: Binary target labels
        selected_drivers_12: List of driver info dicts from Stage 1
        target_final: Final number of features (default: 6)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (final_selected_drivers, cut_drivers, rf_model)
        
    Example:
        >>> final_6, cut_6, rf_model = refine_features_with_rf(
        >>>     X_selected_12, targets_binary, selected_drivers_12, target_final=6
        >>> )
        >>> print(f"Final drivers: {[d['driver_number'] for d in final_6]}")
    """
    
    # Validation
    if len(selected_drivers_12) != 12:
        raise ValueError(f"Expected 12 pre-selected drivers, got {len(selected_drivers_12)}")
    
    if target_final >= len(selected_drivers_12):
        raise ValueError(f"target_final ({target_final}) must be less than 12")
    
    if verbose:
        print(f"\n  STAGE 2: RF REFINEMENT (12 → {target_final})")
        print(f"    Input: {len(selected_drivers_12)} pre-selected features")
        print(f"    Target: {target_final} final features")
        print(f"    Feature-to-sample ratio: {len(X_selected_12)}/{len(selected_drivers_12)} = {len(X_selected_12)/len(selected_drivers_12):.1f}")
        print(f"    Final ratio will be: {len(X_selected_12)}/{target_final} = {len(X_selected_12)/target_final:.1f}")
    
    # Configure Random Forest
    rf_refiner = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        bootstrap=False,
    )
    
    selected_features = [d['feature'] for d in selected_drivers_12]
    
    if verbose:
        print(f"    RF Configuration:")
        print(f"      Trees: {rf_refiner.get_params()['n_estimators']}, Max depth: {rf_refiner.get_params()['max_depth']}")
        print(f"      Features per split: √{len(selected_features)} ≈ {int(np.sqrt(len(selected_features)))}")
    
    # Train Random Forest
    print(f"    Training Random Forest...")
    rf_refiner.fit(X_selected_12, y)
    
    # Extract importance scores
    rf_importances = rf_refiner.feature_importances_
    
    if verbose:
        print(f"    RF Importance range: [{min(rf_importances):.4f}, {max(rf_importances):.4f}]")
        
        # Quality assessment
        importance_std = np.std(rf_importances)
        importance_max = np.max(rf_importances)
        
        if importance_std > 0.05 and importance_max < 0.8:
            print(f"    ✓ Well-distributed importance - high confidence")
        elif importance_std > 0.03:
            print(f"    ✓ Reasonable importance spread")
        else:
            print(f"    ⚠ Consider reviewing feature quality")
    
    # Create enhanced driver rankings
    rf_rankings = []
    for i, driver_info in enumerate(selected_drivers_12):
        enhanced_driver = {
            **driver_info,
            'rf_importance': rf_importances[i],
            'rf_rank': 0,
            'stage1_rank': i + 1,
            'selection_stage': 'RF_evaluated'
        }
        rf_rankings.append(enhanced_driver)
    
    # Sort by RF importance
    rf_rankings.sort(key=lambda x: x['rf_importance'], reverse=True)
    
    # Assign RF ranks
    for i, driver_info in enumerate(rf_rankings):
        driver_info['rf_rank'] = i + 1
    
    # Final selection
    final_selected = rf_rankings[:target_final]
    cut_features = rf_rankings[target_final:]
    
    for driver in final_selected:
        driver['final_selection_status'] = 'SELECTED'
    for driver in cut_features:
        driver['final_selection_status'] = 'ELIMINATED'
    
    if verbose:
        print(f"\n    FINAL SELECTION ({target_final} drivers):")
        for i, driver in enumerate(final_selected, 1):
            stage1_info = f"Stage1 #{driver['stage1_rank']}"
            pillar_short = driver['pillar'][:3]
            print(f"      {i}. Driver {driver['driver_number']:>2} ({pillar_short}) | "
                  f"RF: {driver['rf_importance']:.4f} | {stage1_info}")
        
        print(f"\n    ELIMINATED ({len(cut_features)} drivers):")
        for driver in cut_features:
            print(f"      Driver {driver['driver_number']:>2} ({driver['pillar'][:3]}) | "
                  f"RF: {driver['rf_importance']:.4f}")
        
        # Pillar distribution
        pillar_counts = {}
        for driver in final_selected:
            pillar = driver['pillar']
            pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
        
        print(f"\n    Final pillar distribution: {pillar_counts}")
    
    return final_selected, cut_features, rf_refiner


# =================================================================
# SECTION 3: FRAMEWORK 2 - TEMPORAL VALIDATION (CHAPTER 6)
# =================================================================

def setup_walkforward_parameters_weekly(full_df, config=None):
    """
    Setup 26-week walk-forward analysis parameters.
    
    Configures the sliding window methodology for temporal analysis:
    - Window size: 26 weeks (approximately 6 months)
    - Step size: 13 weeks (quarterly analysis)
    - Ensures sufficient statistical power per window
    
    Args:
        full_df: Complete entity dataset
        config: Framework 2 configuration dict
        
    Returns:
        Tuple of (window_size, step_size)
        
    Example:
        >>> window_size, step_size = setup_walkforward_parameters_weekly(entity_data)
        >>> print(f"Analysis: {window_size}-week windows, {step_size}-week steps")
        Analysis: 26-week windows, 13-week steps
    """
    
    if config is None or not isinstance(config, dict):
        config = DEFAULT_CONFIG.get('framework2', {})
    
    window_weeks = config.get('window_size', 26)
    step_weeks = config.get('step_size', 13)
    min_window_weeks = config.get('min_window_samples', 26)
    max_window_weeks = 78
    
    total_samples = len(full_df)
    
    # Adjust if window too large
    if window_weeks > total_samples * 0.8:
        window_weeks = max(min_window_weeks, int(total_samples * 0.6))
    
    # Adjust step size
    min_step = max(1, total_samples // 20)
    step_weeks = max(min_step, min(step_weeks, window_weeks // 3))
    
    # Apply bounds
    window_size = min(max(min_window_weeks, window_weeks), max_window_weeks)
    step_size = max(1, step_weeks)
    
    print(f"  WALK-FORWARD SETUP:")
    print(f"    Dataset: {total_samples} samples")
    print(f"    Window size: {window_size} weeks")
    print(f"    Step size: {step_size} weeks")
    print(f"    Expected windows: {max(1, (total_samples - window_size) // step_size + 1)}")
    
    return window_size, step_size


def calculate_window_materiality_comprehensive(window_df: pd.DataFrame,
                                             selected_features: List[str],
                                             window_id: int,
                                             config: Optional[Dict] = None) -> Optional[Dict]:
    """
    Calculate comprehensive materiality scores for a single 26-week window.
    
    For each window, this function:
    1. Creates 3-class targets from returns
    2. Calculates MI, correlation, AUC, and F1 scores for each driver
    3. Combines into composite materiality score
    4. Identifies dominant ESG pillar
    
    Args:
        window_df: Time window data (26 weeks)
        selected_features: List of feature names (6 drivers from Framework 1)
        window_id: Unique window identifier
        config: Target creation configuration
        
    Returns:
        Dict with comprehensive window analysis or None if insufficient data
        
    Example:
        >>> window_result = calculate_window_materiality_comprehensive(
        >>>     window_df, selected_features, window_id=1
        >>> )
        >>> print(f"Dominant pillar: {window_result['dominant_pillar']}")
    """
    
    # Extract features
    X_window = window_df[selected_features].fillna(0)
    
    # Calculate returns
    returns = np.log(window_df['close_price'] / window_df['close_price'].shift(1))
    returns = returns[~np.isnan(returns)]
    
    # Early exit if insufficient data
    if len(returns) < 30:
        return None
    
    # Create targets
    targets_3class, metadata_3class, _ = create_three_class_targets_skewness_aware(
        returns, verbose=False, config=config
    )
    
    targets_binary, metadata_binary = create_research_enhanced_binary_targets(
        returns, verbose=False, config=config
    )
    
    # Check class diversity
    if len(np.unique(targets_binary)) < 2 or len(np.unique(targets_3class)) < 2:
        return None
    
    # Align features with targets
    if len(X_window) > len(targets_binary):
        X_window = X_window.iloc[-len(targets_binary):]
    elif len(X_window) < len(targets_binary):
        targets_binary = targets_binary[-len(X_window):]
        targets_3class = targets_3class[-len(X_window):]
    
    try:
        # Calculate Mutual Information
        mi_scores = mutual_info_classif(X_window, targets_3class, random_state=42)
        
        # Calculate per-feature metrics
        feature_correlations = []
        feature_aucs_binary = []
        feature_f1s_3class = []

        for i, feature in enumerate(selected_features):
            # Spearman correlation
            try:
                corr_coef, p_val = spearmanr(  # type: ignore
                    X_window.iloc[:, i].values, 
                    np.array(targets_binary)
                )
                
                corr_value = abs(float(corr_coef))
                
                feature_correlations.append(
                    corr_value if not np.isnan(corr_value) else 0
                )
                
            except Exception:
                feature_correlations.append(0)
                    # Binary AUC
            try:
                binary_auc = roc_auc_score(targets_binary, X_window.iloc[:, i])
                binary_auc = max(binary_auc, 1 - binary_auc)
                feature_aucs_binary.append(binary_auc)
            except:
                feature_aucs_binary.append(0.5)
            
            # 3-class F1 score
            try:
                feature_median = np.median(X_window.iloc[:, i])
                simple_pred = np.where(
                    X_window.iloc[:, i] >= feature_median, 2,
                    np.where(X_window.iloc[:, i] >= np.percentile(X_window.iloc[:, i], 33), 1, 0)
                )
                three_class_f1 = f1_score(targets_3class, simple_pred, average='weighted')
                feature_f1s_3class.append(three_class_f1)
            except:
                feature_f1s_3class.append(0.33)
        
        # Calculate combined scores
        combined_scores = []
        for i in range(len(selected_features)):
            mi_norm = mi_scores[i] / max(mi_scores) if max(mi_scores) > 0 else 0
            corr_norm = feature_correlations[i]
            binary_auc_norm = (feature_aucs_binary[i] - 0.5) * 2
            three_class_f1_norm = feature_f1s_3class[i]
            
            combined = (
                0.30 * mi_norm +           # 30% MI (non-linear relationships)
                0.05 * corr_norm +         # 5% Correlation (minimal - just validation)
                0.25 * binary_auc_norm +   # 25% Binary AUC (directionality)
                0.40 * three_class_f1_norm # 40% F1 (primary performance metric)
            )
            combined_scores.append(combined)
        
        # Create feature rankings
        feature_rankings = []
        for i, feature in enumerate(selected_features):
            driver_num = int(feature.split('_')[1])
            
            feature_rankings.append({
                'feature': feature,
                'driver_number': driver_num,
                'pillar': get_esg_pillar(driver_num),
                'mi_score': mi_scores[i],
                'correlation': feature_correlations[i],
                'binary_auc_score': feature_aucs_binary[i],
                'three_class_f1_score': feature_f1s_3class[i],
                'combined_score': combined_scores[i],
                'rank': 0
            })
        
        # Sort and assign ranks
        feature_rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        for i, feature_info in enumerate(feature_rankings):
            feature_info['rank'] = i + 1
        
        # Pillar-level analysis
        pillar_scores = defaultdict(list)
        for feature_info in feature_rankings:
            pillar_scores[feature_info['pillar']].append(feature_info['combined_score'])
        
        pillar_averages = {pillar: np.mean(scores) for pillar, scores in pillar_scores.items()}
        dominant_pillar = max(pillar_averages.items(), key=lambda x: x[1])[0]
        
        return {
            'window_id': window_id,
            'period_start': window_df['date'].iloc[0],
            'period_end': window_df['date'].iloc[-1],
            'sample_count': len(X_window),
            'target_distribution': {
                'binary': dict(Counter(targets_binary)),
                'three_class': dict(Counter(targets_3class))
            },
            'feature_rankings': feature_rankings,
            'pillar_scores': dict(pillar_averages),
            'dominant_pillar': dominant_pillar,
            'top_driver': feature_rankings[0]['driver_number'] if feature_rankings else None,
            'top_score': feature_rankings[0]['combined_score'] if feature_rankings else 0,
            'target_metadata': {
                'binary': metadata_binary,
                'three_class': metadata_3class
            }
        }
        
    except Exception as e:
        print(f"  Window {window_id} analysis failed: {e}")
        return None


def run_walkforward_analysis_comprehensive_original(full_df: pd.DataFrame,
                                         selected_features: List[str],
                                         config: Optional[Dict] = None) -> Tuple[List[Dict], int, int]:
    """
    Execute Framework 2 walk-forward analysis with 26-week windows.
    
    Implements sliding window methodology:
    - Window size: 26 weeks (6 months)
    - Step size: 13 weeks (quarterly)
    - 1-week gap to prevent look-ahead bias
    
    Process:
    1. Setup 26-week windows with quarterly steps
    2. For each window: calculate materiality scores
    3. Track dominant ESG pillar per window
    4. Return comprehensive temporal analysis
    
    Args:
        full_df: Complete entity dataset
        selected_features: List of 6 features from Framework 1
        config: Framework 2 configuration dict
        
    Returns:
        Tuple of (walkforward_results, window_size, step_size)
        
    Example:
        >>> results, window_size, step_size = run_walkforward_analysis_comprehensive_original(
        >>>     entity_data, selected_features
        >>> )
        >>> print(f"Analyzed {len(results)} windows")
    """
    
    print(f"\n  FRAMEWORK 2: 26-WEEK TEMPORAL VALIDATION")
    
    # Setup parameters
    window_size, step_size = setup_walkforward_parameters_weekly(full_df, config)
    
    # Execute walk-forward
    walkforward_results = []
    window_id = 0
    
    print(f"    Analyzing temporal windows...")
    
    for start_idx in range(window_size + 1, len(full_df) - step_size + 1, step_size):
        window_end = start_idx - 1        # 1-week gap
        window_start = start_idx - window_size - 1
        
        window_df = full_df.iloc[window_start:window_end].copy()
        window_id += 1
        
        # Analyze window
        window_result = calculate_window_materiality_comprehensive(
            window_df, selected_features, window_id, config
        )
        
        if window_result is not None:
            walkforward_results.append(window_result)
            period_str = window_result['period_start'].strftime('%Y-%m')
            top_driver = window_result['feature_rankings'][0]
            print(f"      Window {window_id}: {period_str}")
            print(f"        Dominant: {window_result['dominant_pillar']}")
            print(f"        Top Driver: {top_driver['driver_number']} ({top_driver['pillar']})")
            print(f"        Scores - MI: {top_driver['mi_score']:.3f} | "
                f"AUC: {top_driver['binary_auc_score']:.3f} | "
                f"F1: {top_driver['three_class_f1_score']:.3f} | "
                f"Combined: {top_driver['combined_score']:.3f}")    
    
    if len(walkforward_results) < 2:
        raise ValueError(f"Insufficient valid windows: {len(walkforward_results)}")
    
    print(f"    Completed: {len(walkforward_results)} valid windows")
    
    return walkforward_results, window_size, step_size


# =================================================================
# SECTION 4: REGIME DETECTION (CHAPTER 6)
# =================================================================

def detect_pillar_level_regimes(walkforward_results: List[Dict],
                               config: Optional[Dict] = None) -> Tuple[Dict, List[Dict]]:
    """
    Detect ESG pillar regime changes across temporal windows.
    
    Analyzes when market focus shifts between Environmental, Social, and 
    Governance pillars. A regime change occurs when the dominant pillar
    shifts and remains stable for minimum duration.
    
    Args:
        walkforward_results: Results from Framework 2 walkforward analysis
        config: Regime detection configuration dict
        
    Returns:
        Tuple of (regime_analysis, regimes)
            regime_analysis: Summary statistics
            regimes: List of regime periods with details
            
    Example:
        >>> regime_analysis, regimes = detect_pillar_level_regimes(walkforward_results)
        >>> print(f"Current regime: {regime_analysis['current_regime']}")
        >>> for regime in regimes:
        >>>     print(f"{regime['pillar']}: Windows {regime['regime_start']}-{regime['regime_end']}")
    """
    
    if config is None:
        config = DEFAULT_CONFIG['framework2']
    
    min_regime_length = config.get('regime_min_length', 3)
    
    print(f"  Detecting pillar-level regime changes...")
    
    # Extract dominant pillar sequence
    pillar_sequence = []
    window_info = []
    
    for result in walkforward_results:
        pillar_sequence.append(result['dominant_pillar'])
        window_info.append({
            'window_id': result['window_id'],
            'period_start': result['period_start'],
            'period_end': result['period_end'],
            'dominant_pillar': result['dominant_pillar'],
            'pillar_scores': result['pillar_scores']
        })
    
    # Detect regimes
    regimes = []
    current_regime_pillar = pillar_sequence[0]
    current_regime_start = 0
    regime_count = defaultdict(int)
    
    for i in range(1, len(pillar_sequence)):
        if pillar_sequence[i] != current_regime_pillar:
            # Check if new pillar is stable
            upcoming_windows = pillar_sequence[i:i+min_regime_length]
            if len(upcoming_windows) >= min_regime_length and all(p == pillar_sequence[i] for p in upcoming_windows):
                # End current regime
                regimes.append({
                    'regime_id': len(regimes) + 1,
                    'regime_start': current_regime_start,
                    'regime_end': i - 1,
                    'pillar': current_regime_pillar,
                    'duration_windows': i - current_regime_start,
                    'period_start': window_info[current_regime_start]['period_start'],
                    'period_end': window_info[i - 1]['period_end'],
                    'avg_pillar_scores': calculate_avg_pillar_scores(
                        window_info[current_regime_start:i]
                    )
                })
                regime_count[current_regime_pillar] += 1
                
                # Start new regime
                current_regime_start = i
                current_regime_pillar = pillar_sequence[i]
    
    # Add final regime
    if current_regime_start < len(pillar_sequence):
        regimes.append({
            'regime_id': len(regimes) + 1,
            'regime_start': current_regime_start,
            'regime_end': len(pillar_sequence) - 1,
            'pillar': current_regime_pillar,
            'duration_windows': len(pillar_sequence) - current_regime_start,
            'period_start': window_info[current_regime_start]['period_start'],
            'period_end': window_info[-1]['period_end'],
            'avg_pillar_scores': calculate_avg_pillar_scores(
                window_info[current_regime_start:]
            )
        })
        regime_count[current_regime_pillar] += 1
    
    # Analyze regimes
    regime_analysis = {
        'total_regimes': len(regimes),
        'regime_count_by_pillar': dict(regime_count),
        'average_regime_duration': np.mean([r['duration_windows'] for r in regimes]),
        'dominant_regime_pillar': max(regime_count.items(), key=lambda x: x[1])[0] if regime_count else 'Unknown',
        'regime_stability': 1 - (len(regimes) / len(walkforward_results)),
        'current_regime': regimes[-1]['pillar'] if regimes else 'Unknown'
    }
    
    print(f"    Detected {len(regimes)} pillar regimes")
    print(f"    Current regime: {regime_analysis['current_regime']}")
    print(f"    Regime stability: {regime_analysis['regime_stability']:.3f}")
    
    return regime_analysis, regimes


def track_individual_driver_regimes(walkforward_results: List[Dict],
                                  config: Optional[Dict] = None) -> Dict[int, Dict]:
    """
    Track individual driver regime changes across temporal windows.
    
    Analyzes how individual ESG drivers perform over time, identifying:
    - Improving drivers (rank consistently getting better)
    - Declining drivers (rank consistently getting worse)
    - Stable drivers (consistent performance)
    
    Args:
        walkforward_results: Results from Framework 2 walkforward analysis
        config: Regime detection configuration dict
        
    Returns:
        Dict mapping driver_number to regime analysis
        
    Example:
        >>> driver_regimes = track_individual_driver_regimes(walkforward_results)
        >>> for driver_num, regime_info in driver_regimes.items():
        >>>     print(f"Driver {driver_num}: {regime_info['current_regime']}")
    """
    
    if config is None:
        config = DEFAULT_CONFIG['framework2']
    
    min_regime_length = config.get('regime_min_length', 2)
    rank_threshold = 2  # Rank change threshold
    
    print(f"  Tracking individual driver regimes...")
    
    # Track rank evolution
    driver_rank_evolution = defaultdict(list)
    driver_score_evolution = defaultdict(list)
    
    for result in walkforward_results:
        window_ranks = {}
        window_scores = {}
        
        for feature_info in result['feature_rankings']:
            driver_num = feature_info['driver_number']
            window_ranks[driver_num] = feature_info['rank']
            window_scores[driver_num] = feature_info['combined_score']
        
        # Get all drivers
        all_drivers = set()
        for res in walkforward_results:
            for feat in res['feature_rankings']:
                all_drivers.add(feat['driver_number'])
        
        # Record ranks
        worst_rank = len(all_drivers) + 1
        for driver_num in all_drivers:
            driver_rank_evolution[driver_num].append(window_ranks.get(driver_num, worst_rank))
            driver_score_evolution[driver_num].append(window_scores.get(driver_num, 0))
    
    # Detect regimes for each driver
    driver_regimes = {}
    
    for driver_num, rank_sequence in driver_rank_evolution.items():
        if len(rank_sequence) < min_regime_length * 2:
            continue
        
        regimes = []
        current_regime_start = 0
        current_regime_type = "stable"
        
        for i in range(min_regime_length, len(rank_sequence)):
            # Look at recent trend
            recent_changes = [rank_sequence[j] - rank_sequence[j-1] 
                            for j in range(i-min_regime_length+1, i+1)]
            
            # Determine regime type
            if all(change <= -rank_threshold for change in recent_changes):
                new_regime_type = "improving"  # Lower rank = better
            elif all(change >= rank_threshold for change in recent_changes):
                new_regime_type = "declining"
            else:
                new_regime_type = "stable"
            
            # Check for regime change
            if new_regime_type != current_regime_type and i - current_regime_start >= min_regime_length:
                regimes.append({
                    'regime_start': current_regime_start,
                    'regime_end': i - 1,
                    'regime_type': current_regime_type,
                    'avg_rank': np.mean(rank_sequence[current_regime_start:i]),
                    'duration': i - current_regime_start
                })
                
                current_regime_start = i
                current_regime_type = new_regime_type
        
        # Add final regime
        if current_regime_start < len(rank_sequence):
            regimes.append({
                'regime_start': current_regime_start,
                'regime_end': len(rank_sequence) - 1,
                'regime_type': current_regime_type,
                'avg_rank': np.mean(rank_sequence[current_regime_start:]),
                'duration': len(rank_sequence) - current_regime_start
            })
        
        # Calculate metrics
        driver_regimes[driver_num] = {
            'regimes': regimes,
            'num_regimes': len(regimes),
            'regime_stability': 1 - (len(regimes) / len(rank_sequence)) if len(rank_sequence) > 0 else 1,
            'current_regime': regimes[-1]['regime_type'] if regimes else 'unknown',
            'improvement_periods': len([r for r in regimes if r['regime_type'] == 'improving']),
            'decline_periods': len([r for r in regimes if r['regime_type'] == 'declining']),
            'pillar': get_esg_pillar(driver_num),
            'avg_rank': np.mean(rank_sequence),
            'rank_volatility': np.std(rank_sequence)
        }
    
    print(f"    Completed regime analysis for {len(driver_regimes)} drivers")
    
    # Find most stable drivers
    stable_drivers = sorted(
        [(d, info['regime_stability']) for d, info in driver_regimes.items()],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    print(f"    Most stable drivers: {[d[0] for d in stable_drivers]}")
    
    return driver_regimes


def calculate_driver_stability_metrics(walkforward_results: List[Dict],
                                     individual_regimes: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Calculate comprehensive stability metrics for each driver.
    
    Combines walkforward consistency with regime analysis to provide
    overall stability assessment for each ESG driver.
    
    Args:
        walkforward_results: Results from Framework 2
        individual_regimes: Dict from track_individual_driver_regimes()
        
    Returns:
        Dict mapping driver_number to stability metrics
        
    Example:
        >>> stability = calculate_driver_stability_metrics(results, driver_regimes)
        >>> for driver_num, metrics in stability.items():
        >>>     print(f"Driver {driver_num} stability: {metrics['combined_stability']:.3f}")
    """
    
    print(f"  Calculating driver stability metrics...")
    
    driver_stability = {}
    
    # Get rank and score evolution
    driver_ranks_over_time = defaultdict(list)
    driver_scores_over_time = defaultdict(list)
    
    for result in walkforward_results:
        for feature_info in result['feature_rankings']:
            driver_num = feature_info['driver_number']
            driver_ranks_over_time[driver_num].append(feature_info['rank'])
            driver_scores_over_time[driver_num].append(feature_info['combined_score'])
    
    # Calculate stability for each driver
    for driver_num in driver_ranks_over_time:
        ranks = driver_ranks_over_time[driver_num]
        scores = driver_scores_over_time[driver_num]
        
        if len(ranks) >= 3:
            # Basic statistics
            rank_std = np.std(ranks)
            score_std = np.std(scores)
            avg_rank = np.mean(ranks)
            avg_score = np.mean(scores)
            
            # Stability scores
            rank_stability = 1.0 / (1.0 + rank_std)
            score_stability = 1.0 - min(score_std / max(avg_score, 0.001), 1.0)
            
            ## Trend analysis
            if len(ranks) >= 5:
                x = np.arange(len(ranks))
                
                # Use attribute access (best for scipy 1.15.3)
                rank_result = stats.linregress(x, ranks)
                score_result = stats.linregress(x, scores)
                
                rank_slope = float(rank_result.slope)
                rank_r_value = float(rank_result.rvalue)
                score_slope = float(score_result.slope)
                score_r_value = float(score_result.rvalue)
            else:
                rank_slope = 0.0
                rank_r_value = 0.0
                score_slope = 0.0
                score_r_value = 0.0
            
            # Get regime info
            regime_info = individual_regimes.get(driver_num, {})
            
            # Combined stability score
            combined_stability = (
                0.4 * rank_stability + 
                0.4 * score_stability + 
                0.2 * regime_info.get('regime_stability', 0)
            )
            
            # Store driver stability metrics
            driver_stability[driver_num] = {
                'avg_rank': float(avg_rank),
                'rank_std': float(rank_std),
                'avg_score': float(avg_score),
                'score_std': float(score_std),
                'rank_stability': float(rank_stability),
                'score_stability': float(score_stability),
                'combined_stability': float(combined_stability),
                'rank_trend_slope': rank_slope,
                'score_trend_slope': score_slope,
                'rank_trend_r2': rank_r_value ** 2,
                'score_trend_r2': score_r_value ** 2,
                'windows_analyzed': len(ranks),
                'pillar': get_esg_pillar(driver_num),
                'current_regime': regime_info.get('current_regime', 'unknown'),
                'is_emerging': rank_slope < -0.1 and score_slope > 0.01,
                'is_declining': rank_slope > 0.1 and score_slope < -0.01
            }
    
    print(f"    Stability analysis completed for {len(driver_stability)} drivers")
    
    # Identify emerging drivers
    emerging_drivers = [d for d, info in driver_stability.items() if info['is_emerging']]
    if emerging_drivers:
        print(f"    Emerging drivers detected: {emerging_drivers}")
    
    return driver_stability
# =================================================================
# SECTION 5: UTILITY FUNCTIONS
# =================================================================

def get_esg_pillar(driver_number: int) -> str:
    """
    Map driver number to ESG pillar.
    
    Mapping based on SASB materiality framework:
    - Drivers 1-6: Environmental
    - Drivers 7-18: Social
    - Drivers 19-27: Governance
    
    Args:
        driver_number: PBS driver number (1-27)
        
    Returns:
        ESG pillar name: 'Environmental', 'Social', or 'Governance'
        
    Example:
        >>> pillar = get_esg_pillar(4)
        >>> print(pillar)
        Environmental
    """
    if 1 <= driver_number <= 6:
        return "Environmental"
    elif 7 <= driver_number <= 18:
        return "Social"
    elif 19 <= driver_number <= 27:
        return "Governance"
    else:
        return "Unknown"
    
def get_driver_name(driver_number: int) -> str:
    """
    Map driver number to human-readable name.
    
    Based on SASB materiality framework adapted by Mettle Capital.
    Note: Driver 16 is intentionally omitted from the taxonomy.
    
    Args:
        driver_number: PBS driver number (1-27, excluding 16)
        
    Returns:
        Driver name string
        
    Example:
        >>> name = get_driver_name(3)
        >>> print(name)
        Energy Management
    """
    driver_names = {
        # Environmental (1-6)
        1: "Air Quality",
        2: "Ecological Impacts",
        3: "Energy Management",
        4: "GHG Emissions",
        5: "Waste & Hazardous Materials Management",
        6: "Water & Wastewater Management",
        
        # Social (7-15, 17)
        7: "Access & Affordability",
        8: "Customer Privacy",
        9: "Customer Welfare",
        10: "Data Security",
        11: "Employee Engagement",
        12: "Employee Health and Safety",
        13: "Human Rights & Community Relations",
        14: "Labour Practices",
        15: "Product Quality & Safety",
        16: "Selling Practices & Product Labelling",
        
        # Governance (17-26)
        17: "Business Ethics",
        18: "Business Model Resilience",
        19: "Competitive Behaviour",
        20: "Critical Incident Risk Management",
        21: "Management of the Legal & Regulatory Environment",
        22: "Materials Sourcing & Efficiency",
        23: "Product Design & Lifecycle Management",
        24: "Physical Impacts of Climate Change",
        25: "Supply Chain Management",
        26: "System Risk Management"
    }
    
    if driver_number == 16:
        return "[Reserved]"
    
    return driver_names.get(driver_number, f"Driver {driver_number}")


def calculate_avg_pillar_scores(window_infos: List[Dict]) -> Dict[str, float]:
    """
    Calculate average ESG pillar scores across multiple windows.
    
    Helper function for regime detection that computes average pillar
    performance during regime periods.
    
    Args:
        window_infos: List of window info dictionaries with 'pillar_scores'
        
    Returns:
        Dict mapping pillar names to average scores
        
    Example:
        >>> avg_scores = calculate_avg_pillar_scores(regime_windows)
        >>> print(f"Environmental avg: {avg_scores['Environmental']:.3f}")
    """
    
    pillar_totals = defaultdict(list)
    
    for window in window_infos:
        for pillar, score in window['pillar_scores'].items():
            pillar_totals[pillar].append(score)
    
    return {str(pillar): float(np.mean(scores)) for pillar, scores in pillar_totals.items()}


# =================================================================
# MODULE METADATA
# =================================================================

__version__ = "2.0"
__author__ = "Majid Jangani"
__book__ = "ESG Financial Materiality Assessment"

# Function exports for clean imports
__all__ = [
    # Target Creation
    'create_three_class_targets_skewness_aware',
    'create_research_enhanced_binary_targets',
    
    # Framework 1: Stage 1 (27 → 12)
    'calculate_unified_importance_scores_with_binary',
    'create_enhanced_composite_scores',
    'select_balanced_drivers_enhanced',
    
    # Framework 1: Stage 2 (12 → 6)
    'refine_features_with_rf',
    'setup_conservative_cv',
    
    # Framework 2: Temporal Validation
    'setup_walkforward_parameters_weekly',
    'calculate_window_materiality_comprehensive',
    'run_walkforward_analysis_comprehensive_original',
    
    # Regime Detection
    'detect_pillar_level_regimes',
    'track_individual_driver_regimes',
    'calculate_driver_stability_metrics',
    
    # Utilities
    'get_esg_pillar',
    'get_driver_name',
    'calculate_avg_pillar_scores',

    
    # Configuration
    'DEFAULT_CONFIG'
]