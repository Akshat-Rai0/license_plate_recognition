import sys
from src.pipeline import recognize_plate

def main():
    if len(sys.argv) != 2:
        print("Usage: python recognize_plate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        plate_text = recognize_plate(image_path)
        print(f"Recognized license plate: {plate_text}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()