"""
N-gram Based Evaluation for Cuneiform Sign Detection
"""

import json
import os
import sys
from pathlib import Path
import requests
from typing import Dict, List, Tuple
from ebl_ngrams import FragmentModel
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

# Import broken sign filter — ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing.broken_sign_filter import filter_broken_from_text
from data_processing.sign_resolver import SignToABZResolver


# ============================================================================
# Configuration Section
# ============================================================================

# Path to detection results JSON file
DETECTION_JSON_PATH = os.path.expanduser(
    "~/erc-work-data/signs_detection_on_dataset/eBL_OCRed_Signs_02-19_4-0.6-35.json"
)
DETECTION_JSON_PATH = os.path.expanduser(
    "./output_inference_03-02_try-3-0.8-0.35-2-0.006.json"
)

# Output directory
OUTPUT_DIR = "./evaluation_output_new_line_det_0.8_0.35_2_0.006_20260525-anchor-test"

# Training set directory (images to exclude from evaluation)
TRAIN_SET_DIR = os.path.expanduser(
    "~/erc-work-data/coco-recognitioin-2025-09-25/data-coco/coco/train2017/"
)

import dotenv
dotenv.load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in environment variables. Please set it in your .env file.")



class FragmentModelWithOverlaps(FragmentModel):
    """Extended FragmentModel that returns overlap information"""
    
    def match_with_overlaps(
        self, 
        other: FragmentModel, 
        *n_values, 
        length_weighting=False
    ) -> Tuple[float, dict]:
        """
        Calculate match score and return overlap information
        
        Returns:
            Tuple of (score, overlap_info)
        """
        from ebl_ngrams.metrics import no_weight, weight_by_len
        
        intersection = self.intersection(other, *n_values)
        weighted_sum = weight_by_len if length_weighting else no_weight
        
        intersection_size = weighted_sum(intersection)
        self_size = weighted_sum(self.get_ngrams(*n_values))
        other_size = weighted_sum(other.get_ngrams(*n_values))
        
        score = (
            intersection_size / min(self_size, other_size)
            if self_size and other_size
            else 0.0
        )
        
        overlap_info = {
            'overlap': intersection,
            'overlap_size': len(intersection)
        }
        
        return score, overlap_info


def fetch_all_ground_truth_signs() -> Dict[str, str]:
    """Fetch all ground truth signs from MongoDB with broken sign filtering and ABZ conversion"""
    import time

    def _make_client():
        return MongoClient(MONGODB_URI)

    print("Connecting to MongoDB...")
    client = _make_client()
    db = client['ebl']
    fragments_collection = db['fragments']

    print("Initializing ABZ resolver with cache...")
    abz_resolver = SignToABZResolver(MONGODB_URI)
    abz_resolver.connect()

    print("Fetching fragments from MongoDB...")

    PAGE_SIZE = 5000
    total = fragments_collection.count_documents({'text': {'$exists': True}})

    ground_truth_map = {}
    processed_count = 0
    filtered_count = 0
    last_id = None

    with tqdm(total=total, desc="Processing fragments") as pbar:
        while True:
            # Fetch one page with retry/reconnect on connection drop
            for attempt in range(5):
                try:
                    query = {'text': {'$exists': True}}
                    if last_id is not None:
                        query['_id'] = {'$gt': last_id}
                    batch = list(fragments_collection.find(
                        query, {'_id': 1, 'text': 1}
                    ).sort('_id', 1).limit(PAGE_SIZE))
                    break
                except Exception as e:
                    if attempt < 4:
                        wait = 2 ** attempt
                        tqdm.write(f"\nConnection error (attempt {attempt+1}/5): {e}. Reconnecting in {wait}s...")
                        time.sleep(wait)
                        try:
                            client.close()
                        except Exception:
                            pass
                        client = _make_client()
                        db = client['ebl']
                        fragments_collection = db['fragments']
                    else:
                        raise

            if not batch:
                break
            last_id = batch[-1]['_id']

            for doc in batch:
                fragment_id = doc.get('_id', '')
                text_data = doc.get('text')

                if not fragment_id or not text_data:
                    pbar.update(1)
                    continue

                processed_count += 1

                try:
                    lines = filter_broken_from_text(text_data)
                    if lines:
                        signs_text = ' '.join(' '.join(line) for line in lines)
                        abz_text = abz_resolver.convert_signs_to_abz(signs_text)
                        if abz_text.strip():
                            ground_truth_map[fragment_id] = abz_text
                            filtered_count += 1
                except Exception as e:
                    tqdm.write(f"\nError processing fragment {fragment_id}: {e}")
                pbar.update(1)

    # Output cache statistics
    cache_stats = abz_resolver.get_cache_stats()
    print(f"\nABZ Resolver Cache Stats:")
    print(f"  Cached signs: {cache_stats['cached_signs']}")
    print(f"  Found in DB: {cache_stats['found']}")
    print(f"  Not found: {cache_stats['not_found']}")

    abz_resolver.close()
    client.close()

    print(f"\nProcessed {processed_count} fragments from MongoDB")
    print(f"Successfully filtered {filtered_count} fragments with text data")
    print(f"Loaded {len(ground_truth_map)} fragments with ground truth (ABZ format)\n")

    return ground_truth_map


def load_detection_results(json_file_path: str) -> List[Dict]:
    """Load detection results from JSON file"""
    print(f"Loading detection results from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} detection results\n")
    return data


def load_train_set_fragment_ids(train_dir: str) -> set:
    """Load fragment IDs from training set directory"""
    print(f"Loading training set fragment IDs from: {train_dir}")
    
    if not os.path.exists(train_dir):
        print(f"Warning: Training set directory not found: {train_dir}")
        return set()
    
    train_fragment_ids = set()
    for filename in os.listdir(train_dir):
        if filename.endswith('.jpg'):
            fragment_id = extract_fragment_id(filename)
            train_fragment_ids.add(fragment_id)
    
    print(f"Loaded {len(train_fragment_ids)} training set fragment IDs\n")
    return train_fragment_ids


def extract_fragment_id(filename: str) -> str:
    """Extract fragment ID from filename"""
    return filename.rsplit('.', 1)[0] if '.' in filename else filename


def calculate_match_score(
    detected_signs: str, 
    ground_truth_signs: str,
    fragment_id: str,
    debug: bool = False
) -> Tuple[float, int]:
    """Calculate n-gram match score"""
    if not ground_truth_signs.strip() or not detected_signs.strip():
        return 0.0, 0
    
    ground_truth_doc = FragmentModelWithOverlaps(
        id_=f"{fragment_id}_gt",
        signs=ground_truth_signs,
    )
    ground_truth_doc.set_ngrams()
    
    detected_doc = FragmentModelWithOverlaps(
        id_=f"{fragment_id}_detected",
        signs=detected_signs,
    )
    detected_doc.set_ngrams()
    
    score, overlap_info = detected_doc.match_with_overlaps(ground_truth_doc)
    
    if debug:
        print(f"\n{'='*80}")
        print(f"Debug Fragment: {fragment_id}")
        print(f"{'='*80}")
        print(f"Detected signs: {detected_signs[:200]}")
        print(f"Ground truth signs: {ground_truth_signs[:200]}")
        print(f"\nDetected n-grams: {detected_doc.get_ngrams()}")
        print(f"Ground truth n-grams: {ground_truth_doc.get_ngrams()}")
        print(f"\nIntersection: {overlap_info['overlap']}")
        print(f"Overlap size: {overlap_info['overlap_size']}")
        print(f"Score: {score:.4f}")
        print(f"{'='*80}\n")
    
    return score, overlap_info['overlap_size']


def evaluate_detection_results(
    detection_results: List[Dict],
    ground_truth_map: Dict[str, str],
    train_fragment_ids: set = None
) -> pd.DataFrame:
    """Evaluate all detection results"""
    results = []
    skipped_no_gt = 0  # Count fragments skipped due to missing ground truth
    skipped_train_set = 0  # Count fragments skipped because they are in training set
    
    if train_fragment_ids is None:
        train_fragment_ids = set()
    
    print("Evaluating detection results...")
    for entry in tqdm(detection_results):
        filename = entry.get('filename', '')
        fragment_id = extract_fragment_id(filename)
        detected_signs = entry.get('ocredSigns', '')
        
        # Skip if fragment is in training set
        if fragment_id in train_fragment_ids:
            skipped_train_set += 1
            continue
        
        ground_truth_signs = ground_truth_map.get(fragment_id, '')
        
        if not ground_truth_signs:
            skipped_no_gt += 1
            continue
        
        try:
            score, overlap = calculate_match_score(
                detected_signs, 
                ground_truth_signs, 
                fragment_id
            )
            
            results.append({
                'fragment_id': fragment_id,
                'filename': filename,
                'score': score,
                'overlap': overlap,
                'detected_signs': detected_signs.strip(),
                'ground_truth_signs': ground_truth_signs.strip(),
                'detected_signs_count': len(detected_signs.split()) if detected_signs.strip() else 0,
                'ground_truth_signs_count': len(ground_truth_signs.split()) if ground_truth_signs.strip() else 0,
                'has_detection': bool(detected_signs.strip()),
                'has_ground_truth': bool(ground_truth_signs.strip())
            })
        except Exception as e:
            print(f"Error processing {fragment_id}: {e}")
            continue
    
    # Output skip statistics
    if skipped_train_set > 0:
        print(f"\nSkipped {skipped_train_set} fragments (in training set)")
    if skipped_no_gt > 0:
        print(f"Skipped {skipped_no_gt} fragments (no ground truth available)")
    print(f"Total evaluated: {len(results)} fragments\n")
    
    return pd.DataFrame(results)


# ============================================================================
# Statistics and Visualization
# ============================================================================

def print_evaluation_summary(df: pd.DataFrame):
    """Print evaluation summary statistics"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal fragments evaluated: {len(df)}")
    print(f"Fragments with detections: {df['has_detection'].sum()}")
    print(f"Fragments with ground truth: {df['has_ground_truth'].sum()}")
    
    print(f"\nAverage match score: {df['score'].mean():.4f}")
    print(f"Median match score: {df['score'].median():.4f}")
    print(f"Std deviation: {df['score'].std():.4f}")
    
    print(f"\nAverage overlap size: {df['overlap'].mean():.2f}")
    print(f"Median overlap size: {df['overlap'].median():.2f}")
    
    print("\nOverlap distribution:")
    overlap_bins = [(20, "≥20"), (10, "≥10"), (5, "≥5"), (1, "≥1")]
    for thresh, label in overlap_bins:
        count = (df['overlap'] >= thresh).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count:5d} ({pct:5.1f}%)")
    
    count_zero = (df['overlap'] == 0).sum()
    pct_zero = count_zero / len(df) * 100
    print(f"  =0: {count_zero:5d} ({pct_zero:5.1f}%)")
    
    print("\nScore distribution:")
    thresholds = [(0.8, "≥0.8"), (0.6, "≥0.6"), (0.4, "≥0.4"), (0.2, "≥0.2")]
    for thresh, label in thresholds:
        count = (df['score'] >= thresh).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count:5d} ({pct:5.1f}%)")
    
    count_low = (df['score'] < 0.2).sum()
    pct_low = count_low / len(df) * 100
    print(f"  <0.2: {count_low:5d} ({pct_low:5.1f}%)")
    
    print("\n" + "=" * 80)


def plot_distributions(df: pd.DataFrame, output_path: str):
    """Plot score and overlap distributions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Score histogram
    ax1.hist(df['score'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Match Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Match Scores', fontsize=14)
    ax1.axvline(df['score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["score"].mean():.3f}')
    ax1.axvline(df['score'].median(), color='green', linestyle='--', 
                label=f'Median: {df["score"].median():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score cumulative distribution
    scores_sorted = np.sort(df['score'])
    cumulative = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted)
    ax2.plot(scores_sorted, cumulative, linewidth=2)
    ax2.set_xlabel('Match Score', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution of Match Scores', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add threshold lines
    thresholds = [0.2, 0.4, 0.6, 0.8]
    for thresh in thresholds:
        proportion = (df['score'] >= thresh).mean()
        ax2.axvline(thresh, color='red', linestyle=':', alpha=0.5)
        ax2.text(thresh, 0.5, f'{proportion:.1%}', rotation=90, 
                verticalalignment='center')
    
    # Overlap histogram - detailed 0-100, >100 in single bar
    overlap_data = df['overlap'].copy()
    
    # Plot 0-100 range histogram
    overlap_in_range = overlap_data[overlap_data <= 100]
    ax3.hist(overlap_in_range, bins=50, range=(0, 100), edgecolor='black', alpha=0.7, color='orange')
    
    # Add single bar for >100
    count_over_100 = (overlap_data > 100).sum()
    if count_over_100 > 0:
        # Draw a single bar at x=102 for values >100
        ax3.bar(102, count_over_100, width=4, edgecolor='black', alpha=0.7, color='orange')
        ax3.text(102, count_over_100/2, f'>100\n({count_over_100})', 
                ha='center', va='center', fontsize=9)
    
    ax3.set_xlabel('Overlap Size', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Overlap Sizes (0-100+)', fontsize=14)
    ax3.axvline(df['overlap'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["overlap"].mean():.1f}')
    ax3.axvline(df['overlap'].median(), color='green', linestyle='--', 
                label=f'Median: {df["overlap"].median():.1f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-2, 108)
    
    # Overlap cumulative distribution - show only 0-100
    overlaps_filtered = df[df['overlap'] <= 100]['overlap']
    overlaps_sorted = np.sort(overlaps_filtered)
    cumulative_overlap = np.arange(1, len(overlaps_sorted) + 1) / len(df)  # Use total data as denominator
    ax4.plot(overlaps_sorted, cumulative_overlap, linewidth=2, color='orange')
    ax4.set_xlabel('Overlap Size', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution of Overlap Sizes (0-100)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)
    
    # Add overlap threshold lines
    overlap_thresholds = [1, 5, 10, 20, 50]
    for thresh in overlap_thresholds:
        proportion = (df['overlap'] >= thresh).mean()
        ax4.axvline(thresh, color='red', linestyle=':', alpha=0.5)
        ax4.text(thresh, 0.5, f'{proportion:.1%}', rotation=90, 
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")


def plot_overlap_vs_score(df: pd.DataFrame, output_path: str):
    """Plot relationship between overlap and score"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df['overlap'], df['score'], alpha=0.3, s=10)
    ax.set_xlabel('Overlap Size (number of matching n-grams)', fontsize=12)
    ax.set_ylabel('Match Score', fontsize=12)
    ax.set_title('Relationship between Overlap Size and Match Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(df['overlap'], df['score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['overlap'].min(), df['overlap'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, 
            label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")


def create_detailed_report(df: pd.DataFrame, output_path: str):
    """Generate detailed text report"""
    report = []
    report.append("=" * 80)
    report.append("DETAILED EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS")
    report.append("-" * 80)
    report.append(f"Total fragments evaluated: {len(df)}")
    report.append(f"With detections: {df['has_detection'].sum()}")
    report.append(f"Without detections: {(~df['has_detection']).sum()}")
    report.append("")
    
    # Score statistics
    report.append("SCORE STATISTICS")
    report.append("-" * 80)
    report.append(f"Mean: {df['score'].mean():.4f}")
    report.append(f"Median: {df['score'].median():.4f}")
    report.append(f"Std Dev: {df['score'].std():.4f}")
    report.append(f"Min: {df['score'].min():.4f}")
    report.append(f"Max: {df['score'].max():.4f}")
    report.append(f"25th percentile: {df['score'].quantile(0.25):.4f}")
    report.append(f"75th percentile: {df['score'].quantile(0.75):.4f}")
    report.append("")
    
    # Overlap statistics
    report.append("OVERLAP STATISTICS")
    report.append("-" * 80)
    report.append(f"Mean: {df['overlap'].mean():.2f}")
    report.append(f"Median: {df['overlap'].median():.2f}")
    report.append(f"Std Dev: {df['overlap'].std():.2f}")
    report.append(f"Min: {df['overlap'].min():.0f}")
    report.append(f"Max: {df['overlap'].max():.0f}")
    report.append("")
    
    # Performance distribution
    report.append("PERFORMANCE DISTRIBUTION")
    report.append("-" * 80)
    bins = [
        (0.9, 1.0, "Excellent (0.9-1.0)"),
        (0.8, 0.9, "Very Good (0.8-0.9)"),
        (0.6, 0.8, "Good (0.6-0.8)"),
        (0.4, 0.6, "Fair (0.4-0.6)"),
        (0.2, 0.4, "Poor (0.2-0.4)"),
        (0.0, 0.2, "Very Poor (0.0-0.2)")
    ]
    
    for low, high, label in bins:
        if low == 0.0:
            count = (df['score'] < high).sum()
        elif high == 1.0:
            count = (df['score'] >= low).sum()
        else:
            count = ((df['score'] >= low) & (df['score'] < high)).sum()
        pct = count / len(df) * 100
        report.append(f"{label:35s}: {count:6d} ({pct:5.1f}%)")
    
    report.append("")
    
    # Top 10 best and worst
    report.append("TOP 10 BEST MATCHES")
    report.append("-" * 80)
    top_10 = df.nlargest(10, 'score')[['fragment_id', 'score', 'overlap', 'detected_signs_count', 'ground_truth_signs_count']]
    report.append(top_10.to_string(index=False))
    report.append("")
    
    report.append("TOP 10 WORST MATCHES (with detections)")
    report.append("-" * 80)
    worst_10 = df[df['has_detection']].nsmallest(10, 'score')[
        ['fragment_id', 'score', 'overlap', 'detected_signs_count', 'ground_truth_signs_count']
    ]
    report.append(worst_10.to_string(index=False))
    report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Saved report: {output_path}")


# ============================================================================
# Main Program
# ============================================================================

def main():
    """
    Main evaluation pipeline
    """
    print("\n" + "=" * 80)
    print("Cuneiform Sign Detection Evaluation System")
    print("=" * 80 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load ground truth data
    ground_truth_map = fetch_all_ground_truth_signs()
    
    # Step 2: Load detection results
    detection_results = load_detection_results(DETECTION_JSON_PATH)
    
    # Step 3: Load training set fragment IDs (to exclude from evaluation)
    train_fragment_ids = load_train_set_fragment_ids(TRAIN_SET_DIR)
    
    # Step 4: Evaluate
    df_results = evaluate_detection_results(detection_results, ground_truth_map, train_fragment_ids)
    
    # Step 5: Save detailed results CSV
    csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results: {csv_path}")
    
    # Step 6: Print summary
    print_evaluation_summary(df_results)
    
    # Step 7: Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_distributions(
        df_results, 
        os.path.join(OUTPUT_DIR, 'distributions.png')
    )
    
    plot_overlap_vs_score(
        df_results,
        os.path.join(OUTPUT_DIR, 'overlap_vs_score.png')
    )
    
    # Step 8: Generate text report
    create_detailed_report(
        df_results,
        os.path.join(OUTPUT_DIR, 'detailed_report.txt')
    )
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("  - evaluation_results.csv       (detailed scores)")
    print("  - distributions.png            (score & overlap distributions)")
    print("  - overlap_vs_score.png         (overlap vs score relationship)")
    print("  - detailed_report.txt          (detailed report)")
    print()


if __name__ == "__main__":
    main()
