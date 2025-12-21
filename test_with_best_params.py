import os
import glob
from src.pipeline import recognize_plate

def test_multiple_images(image_dir, num_images=20):
    """Test recognition on multiple images with best parameters"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:num_images]
    
    print("=" * 80)
    print(f"Testing on {len(image_files)} images with optimized parameters")
    print("=" * 80)
    
    results = []
    successful = 0
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            result = recognize_plate(img_path)
            results.append((filename, result, "✓"))
            successful += 1
            print(f"✓ {filename:20s} → {result}")
        except Exception as e:
            results.append((filename, str(e), "✗"))
            error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
            print(f"✗ {filename:20s} → Error: {error_msg}")
    
    print("\n" + "=" * 80)
    print(f"Summary: {successful}/{len(results)} successful ({100*successful/len(results):.1f}%)")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    test_multiple_images("data/raw/car-license-plate-DatasetNinja/ds/img", num_images=20)