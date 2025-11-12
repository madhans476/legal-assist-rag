"""
Retrieval Analytics Dashboard
------------------------------
Analyzes and visualizes retrieval performance metrics.

Features:
- Query type distribution
- Fusion method effectiveness
- Latency analysis
- Weight optimization insights
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, Counter
import statistics


class RetrievalAnalytics:
    """
    Analyzes retrieval logs to provide insights and recommendations.
    """
    
    def __init__(self, analytics_file: Path):
        """
        Initialize analytics from log file.
        
        Args:
            analytics_file: Path to retrieval_analytics.jsonl
        """
        self.analytics_file = analytics_file
        self.metrics = self._load_metrics()
        print(f"[ANALYTICS] Loaded {len(self.metrics)} retrieval records")
    
    def _load_metrics(self) -> List[Dict[str, Any]]:
        """Load metrics from JSONL file."""
        metrics = []
        
        if not self.analytics_file.exists():
            print(f"[WARN] No analytics file found at {self.analytics_file}")
            return metrics
        
        try:
            with open(self.analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
        except Exception as e:
            print(f"[ERROR] Failed to load analytics: {e}")
        
        return metrics
    
    def query_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of query types.
        """
        type_counts = Counter(m['query_type'] for m in self.metrics)
        return dict(type_counts)
    
    def fusion_method_usage(self) -> Dict[str, int]:
        """
        Count how often each fusion method is used.
        """
        fusion_counts = Counter(m['fusion_method'] for m in self.metrics)
        return dict(fusion_counts)
    
    def fusion_by_query_type(self) -> Dict[str, Dict[str, int]]:
        """
        Break down fusion method usage by query type.
        """
        breakdown = defaultdict(lambda: defaultdict(int))
        
        for m in self.metrics:
            qtype = m['query_type']
            fusion = m['fusion_method']
            breakdown[qtype][fusion] += 1
        
        return {k: dict(v) for k, v in breakdown.items()}
    
    def latency_analysis(self) -> Dict[str, Any]:
        """
        Analyze latency patterns.
        """
        if not self.metrics:
            return {}
        
        latencies = [m['total_latency_ms'] for m in self.metrics]
        
        # By fusion method
        by_fusion = defaultdict(list)
        for m in self.metrics:
            by_fusion[m['fusion_method']].append(m['total_latency_ms'])
        
        # By query type
        by_type = defaultdict(list)
        for m in self.metrics:
            by_type[m['query_type']].append(m['total_latency_ms'])
        
        return {
            "overall": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "by_fusion_method": {
                fusion: {
                    "mean": statistics.mean(lats),
                    "median": statistics.median(lats)
                }
                for fusion, lats in by_fusion.items()
            },
            "by_query_type": {
                qtype: {
                    "mean": statistics.mean(lats),
                    "median": statistics.median(lats)
                }
                for qtype, lats in by_type.items()
            }
        }
    
    def weight_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of different weight combinations.
        """
        weight_patterns = defaultdict(list)
        
        for m in self.metrics:
            if m['fusion_method'] == 'weighted':
                weight_key = f"{m['semantic_weight']:.1f}-{m['bm25_weight']:.1f}"
                weight_patterns[weight_key].append({
                    'query_type': m['query_type'],
                    'latency': m['total_latency_ms'],
                    'results': m['results_count']
                })
        
        return dict(weight_patterns)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analytics report.
        """
        if not self.metrics:
            return "No analytics data available."
        
        report = []
        report.append("="*80)
        report.append("RETRIEVAL ANALYTICS REPORT")
        report.append("="*80)
        report.append(f"\nTotal Retrievals: {len(self.metrics)}")
        
        # Query Type Distribution
        report.append("\n" + "-"*80)
        report.append("QUERY TYPE DISTRIBUTION")
        report.append("-"*80)
        type_dist = self.query_type_distribution()
        for qtype, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.metrics)) * 100
            report.append(f"  {qtype:20s} : {count:4d} ({percentage:5.1f}%)")
        
        # Fusion Method Usage
        report.append("\n" + "-"*80)
        report.append("FUSION METHOD USAGE")
        report.append("-"*80)
        fusion_usage = self.fusion_method_usage()
        for method, count in sorted(fusion_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.metrics)) * 100
            report.append(f"  {method:20s} : {count:4d} ({percentage:5.1f}%)")
        
        # Fusion by Query Type
        report.append("\n" + "-"*80)
        report.append("FUSION METHOD BY QUERY TYPE")
        report.append("-"*80)
        fusion_by_type = self.fusion_by_query_type()
        for qtype, methods in fusion_by_type.items():
            report.append(f"\n  {qtype.upper()}:")
            for method, count in methods.items():
                report.append(f"    {method:15s} : {count:3d}")
        
        # Latency Analysis
        report.append("\n" + "-"*80)
        report.append("LATENCY ANALYSIS")
        report.append("-"*80)
        latency = self.latency_analysis()
        
        overall = latency['overall']
        report.append(f"\n  Overall:")
        report.append(f"    Mean   : {overall['mean']:.1f}ms")
        report.append(f"    Median : {overall['median']:.1f}ms")
        report.append(f"    Range  : {overall['min']:.1f}ms - {overall['max']:.1f}ms")
        report.append(f"    StdDev : {overall['stdev']:.1f}ms")
        
        report.append(f"\n  By Fusion Method:")
        for method, stats in latency['by_fusion_method'].items():
            report.append(f"    {method:15s} : {stats['mean']:.1f}ms (median: {stats['median']:.1f}ms)")
        
        report.append(f"\n  By Query Type:")
        for qtype, stats in latency['by_query_type'].items():
            report.append(f"    {qtype:15s} : {stats['mean']:.1f}ms (median: {stats['median']:.1f}ms)")
        
        # Recommendations
        report.append("\n" + "-"*80)
        report.append("RECOMMENDATIONS")
        report.append("-"*80)
        recommendations = self._generate_recommendations(latency, fusion_usage, type_dist)
        for rec in recommendations:
            report.append(f"  • {rec}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def _generate_recommendations(
        self,
        latency: Dict,
        fusion_usage: Dict,
        type_dist: Dict
    ) -> List[str]:
        """
        Generate actionable recommendations based on analytics.
        """
        recommendations = []
        
        # Latency recommendations
        overall_latency = latency['overall']['mean']
        if overall_latency > 500:
            recommendations.append(
                "High average latency detected (>500ms). Consider using 'rrf' or 'weighted' "
                "instead of 'learned' for non-critical queries."
            )
        
        # Check if learned fusion is overused
        if fusion_usage.get('learned', 0) > len(self.metrics) * 0.5:
            recommendations.append(
                "'learned' fusion is used in >50% of queries. Consider using it only for "
                "complex queries (case_based, multi_aspect) to improve speed."
            )
        
        # Check query type patterns
        most_common_type = max(type_dist, key=type_dist.get)
        if most_common_type == 'exact_match':
            recommendations.append(
                "Majority of queries are exact matches. Consider increasing BM25 weight "
                "to 0.6-0.7 for faster and more accurate results."
            )
        elif most_common_type == 'conceptual':
            recommendations.append(
                "Majority of queries are conceptual. Semantic search should be weighted "
                "higher (0.7) for better understanding-based retrieval."
            )
        
        # Fusion method distribution
        if 'rrf' not in fusion_usage and 'weighted' not in fusion_usage:
            recommendations.append(
                "Only using 'learned' fusion. Consider implementing 'rrf' for faster "
                "baseline performance with minimal accuracy loss."
            )
        
        # Collection usage patterns
        multi_collection_queries = sum(
            1 for m in self.metrics if len(m.get('collections_searched', [])) > 1
        )
        if multi_collection_queries > len(self.metrics) * 0.3:
            recommendations.append(
                f"{multi_collection_queries} queries searched multiple collections. "
                "Consider optimizing domain routing threshold for faster single-collection retrieval."
            )
        
        return recommendations
    
    def export_summary(self, output_file: Path):
        """
        Export summary statistics to JSON.
        """
        summary = {
            "total_retrievals": len(self.metrics),
            "query_type_distribution": self.query_type_distribution(),
            "fusion_method_usage": self.fusion_method_usage(),
            "fusion_by_query_type": self.fusion_by_query_type(),
            "latency_analysis": self.latency_analysis(),
            "weight_effectiveness": self.weight_effectiveness()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[EXPORT] Summary saved to {output_file}")


# ===============================================================
# CLI Tool
# ===============================================================
def main():
    """
    Command-line analytics tool.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze retrieval performance")
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path('data/retrieval_analytics.jsonl'),
        help='Path to analytics log file'
    )
    parser.add_argument(
        '--export',
        type=Path,
        help='Export summary to JSON file'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear analytics log after generating report'
    )
    
    args = parser.parse_args()
    
    # Load analytics
    analytics = RetrievalAnalytics(args.log_file)
    
    # Generate and print report
    report = analytics.generate_report()
    print(report)
    
    # Export if requested
    if args.export:
        analytics.export_summary(args.export)
    
    # Clear log if requested
    if args.clear:
        confirm = input("\nClear analytics log? (yes/no): ")
        if confirm.lower() == 'yes':
            args.log_file.unlink()
            print(f"[CLEARED] {args.log_file}")


if __name__ == "__main__":
    main()