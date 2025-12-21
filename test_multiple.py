import os
import glob
from src.pipeline import recognize_plate

def test_multiple_images(image_dir, num_images=10):
    """Test recognition on multiple images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:num_images]
    
    print("=" * 80)
    print(f"Testing on {len(image_files)} images")
    print("=" * 80)
    
    results = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            result = recognize_plate(img_path)
            results.append((filename, result, "✓"))
            print(f"✓ {filename:20s} → {result}")
        except Exception as e:
            results.append((filename, str(e), "✗"))
            print(f"✗ {filename:20s} → Error: {str(e)[:50]}")
    
    print("\n" + "=" * 80)
    successful = sum(1 for _, _, status in results if status == "✓")
    print(f"Summary: {successful}/{len(results)} successful")
    
    return results

if __name__ == "__main__":
    test_multiple_images("data/raw/car-license-plate-DatasetNinja/ds/img", num_images=20)