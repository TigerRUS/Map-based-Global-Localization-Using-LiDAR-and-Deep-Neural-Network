#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_visualize(base_path='dataset/1/input'):
    
    map_path = Path(base_path) / 'map.npz'
    scan_path = Path(base_path) / 'scan.npz'
    
    if map_path.exists():
        print("=" * 50)
        print("LOADING MAP")
        print("=" * 50)
        
        map_data = np.load(map_path)
        map_array = map_data['data']
        width = map_data['width']
        height = map_data['height']
        resolution = map_data['resolution']
        origin_x = map_data['origin_x']
        origin_y = map_data['origin_y']
        
        print(f"Map size: {width}x{height}")
        print(f"Resolution: {resolution} m/pixel")
        print(f"Origin: ({origin_x:.2f}, {origin_y:.2f})")
        print(f"Data shape: {map_array.shape}")
        print(f"Unique values: {np.unique(map_array)}")
        print(f"Free space (-1): {np.sum(map_array == -1)}")
        print(f"Free space (0): {np.sum(map_array == 0)}")
        print(f"Occupied (100): {np.sum(map_array == 100)}")
        
        map_image = np.zeros((height, width, 3), dtype=np.uint8)
        map_image[map_array == -1] = [128, 128, 128]
        map_image[map_array == 0] = [255, 255, 255]
        map_image[map_array == 100] = [0, 0, 0]
    else:
        print(f"Map file not found: {map_path}")
        map_image = None
    
    if scan_path.exists():
        print("\n" + "=" * 50)
        print("LOADING SCAN")
        print("=" * 50)
        
        scan_data = np.load(scan_path)
        scan_array = scan_data['data']
        scan_width = scan_data['width']
        scan_height = scan_data['height']
        
        print(f"Scan map size: {scan_width}x{scan_height}")
        print(f"Scan data shape: {scan_array.shape}")
        print(f"Unique values: {np.unique(scan_array)}")
        print(f"Occupied cells (1): {np.sum(scan_array == 1)}")
        print(f"Free cells (0): {np.sum(scan_array == 0)}")
        
        scan_image = np.zeros((scan_height, scan_width, 3), dtype=np.uint8)
        scan_image[scan_array == 1] = [255, 0, 0]
        scan_image[scan_array == 0] = [255, 255, 255]
    else:
        print(f"Scan file not found: {scan_path}")
        scan_image = None
    
    if map_image is not None or scan_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if map_image is not None:
            axes[0].imshow(map_image, origin='lower')
            axes[0].set_title('Map\n(Gray: unknown, White: free, Black: occupied)')
            axes[0].set_xlabel('X (pixels)')
            axes[0].set_ylabel('Y (pixels)')
        else:
            axes[0].text(0.5, 0.5, 'No map data', ha='center', va='center')
            axes[0].set_title('Map')
        
        if scan_image is not None:
            axes[1].imshow(scan_image, origin='lower')
            axes[1].set_title('Scan in Map Coordinates\n(Red: lidar points, White: free)')
            axes[1].set_xlabel('X (pixels)')
            axes[1].set_ylabel('Y (pixels)')
        else:
            axes[1].text(0.5, 0.5, 'No scan data', ha='center', va='center')
            axes[1].set_title('Scan')
        
        if map_image is not None and scan_image is not None:
            combined = map_image.copy()
            combined[scan_array == 1] = [255, 0, 0]
            
            axes[2].imshow(combined, origin='lower')
            axes[2].set_title('Combined Map + Scan\n(Red: lidar points on map)')
            axes[2].set_xlabel('X (pixels)')
            axes[2].set_ylabel('Y (pixels)')
        else:
            axes[2].text(0.5, 0.5, 'Need both map and scan', ha='center', va='center')
            axes[2].set_title('Combined')
        
        plt.tight_layout()
        plt.show()
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
        
        if map_image is not None:
            binary_map = np.zeros((height, width), dtype=np.uint8)
            binary_map[map_array == 100] = 1
            
            axes2[0].imshow(binary_map, cmap='gray', origin='lower')
            axes2[0].set_title('Binary Map\n(White: occupied, Black: free/unknown)')
            axes2[0].set_xlabel('X (pixels)')
            axes2[0].set_ylabel('Y (pixels)')
        
        if scan_image is not None:
            axes2[1].imshow(scan_array, cmap='gray', origin='lower')
            axes2[1].set_title('Binary Scan\n(White: lidar points, Black: free)')
            axes2[1].set_xlabel('X (pixels)')
            axes2[1].set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.show()
        
        if map_image is not None and scan_image is not None:
            print("\n" + "=" * 50)
            print("COMPARISON STATISTICS")
            print("=" * 50)
            
            map_occupied = (map_array == 100)
            scan_occupied = (scan_array == 1)
            
            overlap = np.sum(map_occupied & scan_occupied)
            map_only = np.sum(map_occupied & ~scan_occupied)
            scan_only = np.sum(~map_occupied & scan_occupied)
            
            print(f"Overlap (both occupied): {overlap} cells")
            print(f"Map only occupied: {map_only} cells")
            print(f"Scan only occupied: {scan_only} cells")
            print(f"Total map occupied: {np.sum(map_occupied)} cells")
            print(f"Total scan occupied: {np.sum(scan_occupied)} cells")
            
            if np.sum(map_occupied) > 0:
                print(f"Overlap/Map ratio: {overlap/np.sum(map_occupied):.2%}")
            if np.sum(scan_occupied) > 0:
                print(f"Overlap/Scan ratio: {overlap/np.sum(scan_occupied):.2%}")

def load_as_matrices(base_path='dataset_np/1/input'):
    """Загружает данные как матрицы для дальнейшей обработки"""
    
    map_path = Path(base_path) / 'map.npz'
    scan_path = Path(base_path) / 'scan.npz'
    
    result = {}
    
    if map_path.exists():
        map_data = np.load(map_path)
        result['map_matrix'] = map_data['data']
        result['map_width'] = map_data['width']
        result['map_height'] = map_data['height']
        result['map_resolution'] = map_data['resolution']
    
    if scan_path.exists():
        scan_data = np.load(scan_path)
        result['scan_matrix'] = scan_data['data']
        result['scan_width'] = scan_data['width']
        result['scan_height'] = scan_data['height']
    
    return result

if __name__ == '__main__':
    load_and_visualize('dataset/12/input')
