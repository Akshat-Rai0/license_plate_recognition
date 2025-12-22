#!/usr/bin/env python3
"""
Diagnose contour detection to understand why plate detection is failing
"""
import sys
import cv2
import shutil
import os

import numpy as np
from pathlib import Path
from src.io_utils import load_image
from src.preprocess import resize_image, convert_to_grayscale, enhance_contrast, reduce_noise

def diagnose_contours(image_path: str):
    """Visualize all contours and their properties"""
    
    print(f"\nDiagnosing: {image_path}\n")
    
    OUTPUT_DIR = "debug_output"
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load and preprocess
    image = load_image(image_path)
    resized = resize_image(image)
    
    # Enhance
    gray_pre = convert_to_grayscale(resized)
    enhanced = enhance_contrast(gray_pre, alpha=1.3, beta=10)
    denoised = reduce_noise(enhanced, method='bilateral')
    resized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    gray = convert_to_grayscale(resized)
    
    # Edge detection with current params
    blur_kernel = (3, 3)
    blurred = cv2.GaussianBlur(gray, blur_kernel, sigmaX=0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Save edges
    cv2.imwrite("debug_output/diag_edges.png", edges)
    print(f"Edges saved to debug_output/diag_edges.png")
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")
    
    # Filter by area
    min_area = 300
    filtered = [c for c in contours if min_area < cv2.contourArea(c) < 50000]
    filtered.sort(key=cv2.contourArea, reverse=True)
    
    print(f"Contours after area filter (>{min_area}): {len(filtered)}\n")
    
    # Analyze top 10 contours
    print("Top 10 contours by area:")
    print(f"{'Rank':<6} {'Area':<10} {'BBox (x,y,w,h)':<25} {'AR':<8} {'Passes?'}")
    print("-" * 70)
    
    for i, contour in enumerate(filtered[:10]):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        ar = w / float(h) if h > 0 else 0
        
        # Check if passes current filter (ar between 2.0 and 4.0)
        passes = "YES" if 2.0 <= ar <= 4.0 else "NO"
        
        print(f"{i+1:<6} {area:<10.0f} ({x},{y},{w},{h}){' '*(25-len(f'({x},{y},{w},{h})'))} {ar:<8.2f} {passes}")
        
        # Draw this contour
        vis = resized.copy()
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(vis, f"#{i+1} AR:{ar:.2f} Area:{area:.0f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(f"debug_output/diag_contour_{i+1}.png", vis)
    
    # Draw all filtered contours
    all_vis = resized.copy()
    cv2.drawContours(all_vis, filtered[:10], -1, (0, 255, 0), 2)
    for i, contour in enumerate(filtered[:10]):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(all_vis, str(i+1), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite("debug_output/diag_all_contours.png", all_vis)
    
    print(f"\nVisualization saved to debug_output/")
    print(f"- diag_edges.png: Edge detection output")
    print(f"- diag_all_contours.png: All top 10 contours")
    print(f"- diag_contour_N.png: Individual contours")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_contours.py <image_path>")
        sys.exit(1)
    
    diagnose_contours(sys.argv[1])
