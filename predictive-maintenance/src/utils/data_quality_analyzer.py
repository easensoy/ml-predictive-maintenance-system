import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    def __init__(self):
        self.quality_thresholds = {
            'missing_data_threshold': 0.05,
            'outlier_threshold': 3.0,
            'variance_threshold': 0.001,
            'correlation_threshold': 0.95
        }
        self.feature_ranges = {
            'vibration_rms': (0.1, 5.0),
            'temperature_bearing': (20, 120),
            'pressure_oil': (5, 30),
            'rpm': (500, 2000),
            'oil_quality_index': (0, 100),
            'power_consumption': (10, 100)
        }
        
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'quality_score': 100,
            'issues': [],
            'feature_analysis': {}
        }
        
        missing_data_analysis = self._analyze_missing_data(data)
        outlier_analysis = self._analyze_outliers(data)
        distribution_analysis = self._analyze_distributions(data)
        correlation_analysis = self._analyze_correlations(data)
        
        quality_report['missing_data'] = missing_data_analysis
        quality_report['outliers'] = outlier_analysis
        quality_report['distributions'] = distribution_analysis
        quality_report['correlations'] = correlation_analysis
        
        quality_score = self._calculate_quality_score(
            missing_data_analysis, outlier_analysis, 
            distribution_analysis, correlation_analysis
        )
        quality_report['quality_score'] = quality_score
        
        if quality_score < 70:
            quality_report['issues'].append('Overall data quality below acceptable threshold')
        
        return quality_report
    
    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict:
        missing_analysis = {
            'total_missing': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'features_with_missing': {},
            'quality_impact': 'LOW'
        }
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(data)) * 100
                missing_analysis['features_with_missing'][column] = {
                    'count': int(missing_count),
                    'percentage': missing_percentage
                }
                
                if missing_percentage > self.quality_thresholds['missing_data_threshold'] * 100:
                    missing_analysis['quality_impact'] = 'HIGH'
        
        return missing_analysis
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict:
        outlier_analysis = {
            'total_outliers': 0,
            'outlier_percentage': 0,
            'features_with_outliers': {},
            'quality_impact': 'LOW'
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        
        for column in numeric_columns:
            if column in data.columns:
                outliers = self._detect_outliers_zscore(data[column])
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(data)) * 100
                    outlier_analysis['features_with_outliers'][column] = {
                        'count': outlier_count,
                        'percentage': outlier_percentage,
                        'outlier_indices': outliers.tolist()
                    }
                    total_outliers += outlier_count
        
        outlier_analysis['total_outliers'] = total_outliers
        outlier_analysis['outlier_percentage'] = (total_outliers / (len(data) * len(numeric_columns))) * 100
        
        if outlier_analysis['outlier_percentage'] > 5:
            outlier_analysis['quality_impact'] = 'HIGH'
        elif outlier_analysis['outlier_percentage'] > 2:
            outlier_analysis['quality_impact'] = 'MEDIUM'
        
        return outlier_analysis
    
    def _detect_outliers_zscore(self, series: pd.Series) -> np.ndarray:
        if series.dtype in ['object', 'string']:
            return np.array([])
        
        series_clean = series.dropna()
        if len(series_clean) < 3:
            return np.array([])
        
        z_scores = np.abs(stats.zscore(series_clean))
        outlier_mask = z_scores > self.quality_thresholds['outlier_threshold']
        
        return series_clean.index[outlier_mask].values
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict:
        distribution_analysis = {
            'feature_distributions': {},
            'normality_tests': {},
            'range_violations': {},
            'quality_impact': 'LOW'
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        range_violations = 0
        
        for column in numeric_columns:
            if column in data.columns and not data[column].empty:
                series_clean = data[column].dropna()
                
                if len(series_clean) > 3:
                    distribution_analysis['feature_distributions'][column] = {
                        'mean': float(series_clean.mean()),
                        'std': float(series_clean.std()),
                        'min': float(series_clean.min()),
                        'max': float(series_clean.max()),
                        'skewness': float(series_clean.skew()),
                        'kurtosis': float(series_clean.kurtosis())
                    }
                    
                    if len(series_clean) > 8:
                        try:
                            shapiro_stat, shapiro_p = stats.shapiro(series_clean.sample(min(len(series_clean), 5000)))
                            distribution_analysis['normality_tests'][column] = {
                                'shapiro_statistic': float(shapiro_stat),
                                'shapiro_p_value': float(shapiro_p),
                                'is_normal': shapiro_p > 0.05
                            }
                        except Exception:
                            distribution_analysis['normality_tests'][column] = {'error': 'Could not perform normality test'}
                    
                    if column in self.feature_ranges:
                        min_val, max_val = self.feature_ranges[column]
                        violations = ((series_clean < min_val) | (series_clean > max_val)).sum()
                        if violations > 0:
                            violation_percentage = (violations / len(series_clean)) * 100
                            distribution_analysis['range_violations'][column] = {
                                'count': int(violations),
                                'percentage': violation_percentage,
                                'expected_range': [min_val, max_val],
                                'actual_range': [float(series_clean.min()), float(series_clean.max())]
                            }
                            range_violations += violations
        
        if range_violations > len(data) * 0.1:
            distribution_analysis['quality_impact'] = 'HIGH'
        elif range_violations > len(data) * 0.05:
            distribution_analysis['quality_impact'] = 'MEDIUM'
        
        return distribution_analysis
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict:
        correlation_analysis = {
            'correlation_matrix': {},
            'high_correlations': [],
            'multicollinearity_detected': False,
            'quality_impact': 'LOW'
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            correlation_analysis['correlation_matrix'] = corr_matrix.round(3).to_dict()
            
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.quality_thresholds['correlation_threshold']:
                        high_corr_pairs.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': round(corr_value, 3)
                        })
            
            correlation_analysis['high_correlations'] = high_corr_pairs
            
            if len(high_corr_pairs) > 0:
                correlation_analysis['multicollinearity_detected'] = True
                correlation_analysis['quality_impact'] = 'MEDIUM'
        
        return correlation_analysis
    
    def _calculate_quality_score(self, missing_analysis: Dict, outlier_analysis: Dict, 
                                distribution_analysis: Dict, correlation_analysis: Dict) -> float:
        score = 100
        
        if missing_analysis['missing_percentage'] > 5:
            score -= 20
        elif missing_analysis['missing_percentage'] > 1:
            score -= 10
        
        if outlier_analysis['outlier_percentage'] > 10:
            score -= 25
        elif outlier_analysis['outlier_percentage'] > 5:
            score -= 15
        elif outlier_analysis['outlier_percentage'] > 2:
            score -= 5
        
        range_violations = len(distribution_analysis.get('range_violations', {}))
        if range_violations > 2:
            score -= 20
        elif range_violations > 0:
            score -= 10
        
        if correlation_analysis.get('multicollinearity_detected', False):
            score -= 15
        
        return max(0, score)
    
    def get_quality_recommendations(self, quality_report: Dict) -> List[str]:
        recommendations = []
        
        if quality_report['missing_data']['missing_percentage'] > 1:
            recommendations.append("Consider data imputation or collection process improvements for missing values")
        
        if quality_report['outliers']['outlier_percentage'] > 5:
            recommendations.append("Investigate and potentially filter outliers in the dataset")
        
        if quality_report['correlations'].get('multicollinearity_detected', False):
            recommendations.append("Remove highly correlated features to avoid multicollinearity")
        
        if len(quality_report['distributions'].get('range_violations', {})) > 0:
            recommendations.append("Validate sensor calibration for features with range violations")
        
        if quality_report['quality_score'] < 70:
            recommendations.append("Overall data quality requires immediate attention before model training")
        
        return recommendations

data_quality_analyzer = DataQualityAnalyzer()