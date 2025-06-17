import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataDriftDetector:
    
    def __init__(self, reference_window_size: int = 1000):
        self.reference_window_size = reference_window_size
        self.reference_data = {}
        self.drift_threshold = 0.05
        self.feature_names = [
            'vibration_rms', 'temperature_bearing', 'pressure_oil',
            'rpm', 'oil_quality_index', 'power_consumption'
        ]
    
    def set_reference_data(self, data: np.ndarray, feature_names: List[str] = None):
        if feature_names:
            self.feature_names = feature_names
        
        for i, feature in enumerate(self.feature_names):
            if i < data.shape[1]:
                feature_data = data[:, i]
                self.reference_data[feature] = {
                    'mean': np.mean(feature_data),
                    'std': np.std(feature_data),
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'distribution': feature_data[-self.reference_window_size:]
                }
    
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, any]:
        if not self.reference_data:
            return {'error': 'No reference data set'}
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'features': {}
        }
        
        for i, feature in enumerate(self.feature_names):
            if i >= new_data.shape[1] or feature not in self.reference_data:
                continue
            
            feature_data = new_data[:, i]
            ref_stats = self.reference_data[feature]
            
            current_mean = np.mean(feature_data)
            current_std = np.std(feature_data)
            
            feature_results = {
                'drift_detected': False,
                'tests': {}
            }
            
            if len(feature_data) > 30:
                t_stat, t_p_value = stats.ttest_1samp(feature_data, ref_stats['mean'])
                feature_results['tests']['mean_shift'] = {
                    'p_value': t_p_value,
                    'drift_detected': t_p_value < self.drift_threshold,
                    'current_mean': current_mean,
                    'reference_mean': ref_stats['mean'],
                    'change_percent': ((current_mean - ref_stats['mean']) / ref_stats['mean']) * 100
                }
            
            if current_std > 0 and ref_stats['std'] > 0:
                f_stat = (current_std ** 2) / (ref_stats['std'] ** 2)
                variance_drift = abs(f_stat - 1) > 0.5
                feature_results['tests']['variance_change'] = {
                    'f_statistic': f_stat,
                    'drift_detected': variance_drift,
                    'current_std': current_std,
                    'reference_std': ref_stats['std']
                }
            
            if len(feature_data) > 30:
                ks_stat, ks_p_value = stats.ks_2samp(
                    ref_stats['distribution'], feature_data
                )
                feature_results['tests']['distribution_drift'] = {
                    'ks_statistic': ks_stat,
                    'p_value': ks_p_value,
                    'drift_detected': ks_p_value < self.drift_threshold
                }
            
            current_min, current_max = np.min(feature_data), np.max(feature_data)
            range_expansion = (
                current_min < ref_stats['min'] * 0.8 or 
                current_max > ref_stats['max'] * 1.2
            )
            feature_results['tests']['range_drift'] = {
                'drift_detected': range_expansion,
                'current_range': [current_min, current_max],
                'reference_range': [ref_stats['min'], ref_stats['max']]
            }
            
            feature_drift = any(
                test.get('drift_detected', False) 
                for test in feature_results['tests'].values()
            )
            feature_results['drift_detected'] = feature_drift
            
            drift_results['features'][feature] = feature_results
            
            if feature_drift:
                drift_results['overall_drift_detected'] = True
        
        if drift_results['overall_drift_detected']:
            drifted_features = [
                feature for feature, results in drift_results['features'].items()
                if results['drift_detected']
            ]
            logger.warning(f"Data drift detected in features: {drifted_features}")
        
        return drift_results
    
    def get_drift_summary(self, drift_results: Dict) -> str:
        if not drift_results.get('overall_drift_detected', False):
            return "No significant data drift detected."
        
        summary_parts = ["Data drift detected:"]
        
        for feature, results in drift_results['features'].items():
            if results['drift_detected']:
                drift_types = [
                    test_name for test_name, test_results in results['tests'].items()
                    if test_results.get('drift_detected', False)
                ]
                summary_parts.append(f"- {feature}: {', '.join(drift_types)}")
        
        return "\n".join(summary_parts)
    
    def update_reference_data(self, new_data: np.ndarray, adaptive: bool = True):
        if not adaptive or not self.reference_data:
            return
        
        alpha = 0.1
        
        for i, feature in enumerate(self.feature_names):
            if i >= new_data.shape[1] or feature not in self.reference_data:
                continue
            
            feature_data = new_data[:, i]
            current_mean = np.mean(feature_data)
            current_std = np.std(feature_data)
            
            self.reference_data[feature]['mean'] = (
                (1 - alpha) * self.reference_data[feature]['mean'] + 
                alpha * current_mean
            )
            
            self.reference_data[feature]['std'] = (
                (1 - alpha) * self.reference_data[feature]['std'] + 
                alpha * current_std
            )
            
            current_distribution = self.reference_data[feature]['distribution']
            new_sample_size = min(len(feature_data), self.reference_window_size // 4)
            
            if len(current_distribution) + new_sample_size > self.reference_window_size:
                samples_to_remove = len(current_distribution) + new_sample_size - self.reference_window_size
                current_distribution = current_distribution[samples_to_remove:]
            
            new_samples = np.random.choice(feature_data, new_sample_size, replace=False)
            self.reference_data[feature]['distribution'] = np.concatenate([
                current_distribution, new_samples
            ])
