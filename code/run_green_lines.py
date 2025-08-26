#!/usr/bin/env python3
"""
Simple script to run palm line detection with green lines on original image.
Usage: python3 run_green_lines.py <input_image>
Example: python3 run_green_lines.py hand1.jpg
"""

import sys
import os
from read_palm import main

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 run_green_lines.py <input_image>")
        print("Example: python3 run_green_lines.py hand1.jpg")
        sys.exit(1)
    
    input_image = sys.argv[1]
    
    # Check if input file exists
    if not os.path.exists(f'input/{input_image}'):
        print(f"Error: Input file 'input/{input_image}' not found!")
        print("Available images in input/ directory:")
        for file in os.listdir('input'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {file}")
        sys.exit(1)
    
    print(f"Processing {input_image}...")
    main(input_image)
    print(f"Result saved as results/result.jpg")
    print("The image shows the original palm with all detected lines in green color.")
