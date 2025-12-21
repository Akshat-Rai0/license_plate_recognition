import itertools
import json
from src.pipeline_tunable import recognize_plate_tunable
from src.io_utils import load_image

# Ground truth
GROUND_TRUTH = "KL01CA2555"
TEST_IMAGE = "data/raw/car-license-plate-DatasetNinja/ds/img/Cars0.png"

# Reduced parameter grid (2,304 combinations - ~7-8 minutes)
PARAM_GRID = {
    'blur_kernel': [(3, 3), (5, 5)],
    'canny_threshold1': [50, 100],
    'canny_threshold2': [150, 200],
    'min_area': [300, 500],
    'ar_min': [2.0, 2.5],
    'ar_max': [4.0, 5.0],
    'binarize_block_size': [9, 11, 13],
    'binarize_C': [10, 12],
    'morph_kernel_size': [(2, 2), (3, 3)],
    'char_min_area': [50, 100],
    'height_ratio_min': [0.3, 0.4],
    'height_ratio_max': [0.85, 0.95],
    'aspect_ratio_min': [0.15, 0.2],
    'aspect_ratio_max': [0.8, 1.0]
}

def calculate_accuracy(predicted, ground_truth):
    """Calculate character-level and exact match accuracy"""
    if len(predicted) != len(ground_truth):
        return 0.0, False
    
    correct = sum(p == g for p, g in zip(predicted, ground_truth))
    char_accuracy = correct / len(ground_truth)
    exact_match = predicted == ground_truth
    
    return char_accuracy, exact_match

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def evaluate_params(params):
    """Evaluate a parameter set"""
    try:
        result = recognize_plate_tunable(TEST_IMAGE, params)
        char_acc, exact = calculate_accuracy(result, GROUND_TRUTH)
        distance = levenshtein_distance(result, GROUND_TRUTH)
        
        return {
            'params': params,
            'predicted': result,
            'char_accuracy': char_acc,
            'exact_match': exact,
            'levenshtein': distance,
            'success': True
        }
    except Exception as e:
        return {
            'params': params,
            'error': str(e),
            'success': False
        }

def grid_search():
    """Run reduced grid search"""
    print("=" * 80)
    print(f"Reduced Grid Search for License Plate Recognition")
    print(f"Ground Truth: {GROUND_TRUTH}")
    print(f"Test Image: {TEST_IMAGE}")
    print("=" * 80)
    
    # Generate all parameter combinations
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[key] for key in keys]
    
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    print(f"\nTotal parameter combinations: {total_combinations:,}")
    print("Estimated time: ~7-8 minutes")
    print("Starting grid search...\n")
    
    best_result = None
    best_score = -1
    results = []
    
    count = 0
    for param_combo in itertools.product(*values):
        count += 1
        params = dict(zip(keys, param_combo))
        
        if count % 100 == 0:
            progress = 100 * count / total_combinations
            print(f"Progress: {count}/{total_combinations} ({progress:.1f}%)")
        
        result = evaluate_params(params)
        results.append(result)
        
        if result['success']:
            # Score: prioritize exact match, then char accuracy, then low Levenshtein distance
            score = (100 if result['exact_match'] else 0) + \
                    result['char_accuracy'] * 50 - \
                    result['levenshtein'] * 5
            
            if score > best_score:
                best_score = score
                best_result = result
                
                print(f"\n{'='*80}")
                print(f"NEW BEST RESULT (Score: {score:.2f})")
                print(f"Predicted: {result['predicted']}")
                print(f"Char Accuracy: {result['char_accuracy']*100:.1f}%")
                print(f"Exact Match: {result['exact_match']}")
                print(f"Levenshtein Distance: {result['levenshtein']}")
                print(f"Parameters:")
                for k, v in params.items():
                    print(f"  {k}: {v}")
                print(f"{'='*80}\n")
                
                if result['exact_match']:
                    print("🎉 EXACT MATCH FOUND! Stopping search...")
                    break
    
    # Summary
    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['success']]
    exact_matches = [r for r in successful if r.get('exact_match', False)]
    
    print(f"Total combinations tested: {len(results)}")
    print(f"Successful runs: {len(successful)}")
    print(f"Exact matches: {len(exact_matches)}")
    
    if best_result:
        print(f"\nBest Result:")
        print(f"  Predicted: {best_result['predicted']}")
        print(f"  Char Accuracy: {best_result['char_accuracy']*100:.1f}%")
        print(f"  Exact Match: {best_result['exact_match']}")
        print(f"  Levenshtein Distance: {best_result['levenshtein']}")
        print(f"\nBest Parameters:")
        for k, v in best_result['params'].items():
            print(f"  {k}: {v}")
        
        # Save best parameters
        with open('best_params.json', 'w') as f:
            json.dump(best_result['params'], f, indent=2)
        print(f"\n✓ Best parameters saved to best_params.json")
    
    # Top 10 results
    successful_sorted = sorted(successful, 
                              key=lambda x: (x['exact_match'], x['char_accuracy'], -x['levenshtein']),
                              reverse=True)
    
    print(f"\nTop 10 Results:")
    for i, r in enumerate(successful_sorted[:10], 1):
        print(f"{i}. {r['predicted']} | Acc: {r['char_accuracy']*100:.1f}% | "
              f"Exact: {r['exact_match']} | Dist: {r['levenshtein']}")

if __name__ == "__main__":
    grid_search()