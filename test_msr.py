#!/usr/bin/env python3
"""
Test script for Multi-Scale Retinex (MSR) implementation
"""

import cv2
import numpy as np
import os
import sys

# Add src directory to path to import enhancement functions
sys.path.append('src')

try:
    from enhance import multi_scale_retinex, retinex_enhancement
    print("✓ Successfully imported MSR functions from enhance.py")
except ImportError as e:
    print(f"✗ Failed to import from enhance.py: {e}")
    sys.exit(1)

def test_msr_function():
    """Test the MSR function with a synthetic dark image"""
    print("\n=== Testing MSR Function ===")
    
    # Create a synthetic dark image for testing
    height, width = 100, 100
    
    # Create a gradient image that's artificially darkened
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    gradient = ((x + y) / 2 * 255).astype(np.uint8)
    
    # Make it darker to simulate low-light conditions
    dark_image = (gradient * 0.3).astype(np.uint8)
    dark_image = cv2.cvtColor(dark_image, cv2.COLOR_GRAY2BGR)
    
    print(f"Created synthetic dark image: {dark_image.shape}")
    print(f"Image value range: {dark_image.min()} - {dark_image.max()}")
    
    try:
        # Test MSR enhancement
        msr_result = multi_scale_retinex(dark_image)
        print(f"✓ MSR enhancement successful")
        print(f"  Input range: {dark_image.min()} - {dark_image.max()}")
        print(f"  MSR output range: {msr_result.min()} - {msr_result.max()}")
        print(f"  Output shape: {msr_result.shape}")
        print(f"  Output dtype: {msr_result.dtype}")
        
        # Test with custom scales and weights
        custom_scales = [10, 50, 200]
        custom_weights = [0.5, 0.3, 0.2]
        msr_custom = multi_scale_retinex(dark_image, scales=custom_scales, weights=custom_weights)
        print(f"✓ MSR with custom parameters successful")
        print(f"  Custom MSR output range: {msr_custom.min()} - {msr_custom.max()}")
        
        # Compare with single scale retinex
        ssr_result = retinex_enhancement(dark_image)
        print(f"✓ Single Scale Retinex comparison")
        print(f"  SSR output range: {ssr_result.min()} - {ssr_result.max()}")
        
        # Save test results if enhanced_images directory exists
        if os.path.exists('enhanced_images'):
            cv2.imwrite('enhanced_images/test_original.jpg', dark_image)
            cv2.imwrite('enhanced_images/test_msr.jpg', msr_result)
            cv2.imwrite('enhanced_images/test_msr_custom.jpg', msr_custom)
            cv2.imwrite('enhanced_images/test_ssr.jpg', ssr_result)
            print("✓ Test images saved to enhanced_images/")
        
        return True
        
    except Exception as e:
        print(f"✗ MSR test failed: {e}")
        return False

def test_edge_cases():
    """Test MSR with edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    try:
        # Test with different image sizes
        small_img = np.ones((10, 10, 3), dtype=np.uint8) * 50
        large_img = np.ones((500, 500, 3), dtype=np.uint8) * 50
        
        msr_small = multi_scale_retinex(small_img)
        msr_large = multi_scale_retinex(large_img)
        print("✓ Different image sizes handled correctly")
        
        # Test with mismatched scales and weights
        try:
            bad_msr = multi_scale_retinex(small_img, scales=[15, 80], weights=[0.5, 0.3, 0.2])
            print("✗ Should have raised ValueError for mismatched scales/weights")
            return False
        except ValueError:
            print("✓ Correctly caught ValueError for mismatched parameters")
        
        # Test with extreme values
        bright_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        dark_img = np.ones((50, 50, 3), dtype=np.uint8) * 5
        
        msr_bright = multi_scale_retinex(bright_img)
        msr_dark = multi_scale_retinex(dark_img)
        print("✓ Extreme brightness values handled correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def main():
    """Run all MSR tests"""
    print("Multi-Scale Retinex (MSR) Test Suite")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('src'):
        print("✗ Please run this script from the project root directory")
        return
    
    test_results = []
    
    # Run function test
    test_results.append(test_msr_function())
    
    # Run edge case tests
    test_results.append(test_edge_cases())
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    if all(test_results):
        print("✓ All tests passed! MSR implementation is working correctly.")
        print("\nThe Multi-Scale Retinex enhancement has been successfully added to your project.")
        print("\nYou can now run:")
        print("  python src/enhance.py        # For the enhanced version with MSR")
        print("  python src/object_detection.py  # For the original version with MSR added")
    else:
        print("✗ Some tests failed. Please check the implementation.")
        
    print("\nMSR Features:")
    print("- Combines multiple scales (15, 80, 250) for better illumination normalization")
    print("- Better color constancy compared to single-scale Retinex")
    print("- Improved performance in varying lighting conditions")
    print("- Configurable scales and weights for different scenarios")

if __name__ == "__main__":
    main()