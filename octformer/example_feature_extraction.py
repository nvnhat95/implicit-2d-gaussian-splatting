#!/usr/bin/env python3
# --------------------------------------------------------
# Example: Extract backbone and relative position features for 3D points
# --------------------------------------------------------

import torch
import numpy as np
from extract_features import FeatureExtractor


def example_single_point_feature_extraction():
    """Example of extracting voxel and relative position features for a single point with context."""
    
    # Configuration
    checkpoint_path = "checkpoints/octformer_scannet200/best_model.pth"  # Use the provided checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create feature extractor
    print("Loading model...")
    extractor = FeatureExtractor(
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Define a single point with NDFP format (Normal 3 + Displacement 1 + Color 3 + Position 3)
    single_point = torch.tensor([0.0, 0.0, 1.0,  # Normal (x, y, z)
                                0.1,                # Displacement
                                0.8, 0.6, 0.4,     # Color (r, g, b)
                                0.5, 0.3, 0.7], dtype=torch.float32)  # Position (x, y, z)
    
    # Create context points around the single point
    num_context_points = 500
    context_points = torch.randn(num_context_points, 10, dtype=torch.float32)
    
    # Center context points around the single point
    context_points[:, 7:10] = context_points[:, 7:10] * 0.2 + single_point[7:10]  # Position
    context_points[:, 0:3] = torch.randn(num_context_points, 3) * 0.1  # Normal
    context_points[:, 3] = torch.randn(num_context_points) * 0.1  # Displacement
    context_points[:, 4:7] = torch.ones(num_context_points, 3) * 0.5  # Color
    
    # Combine into one NDFP tensor
    all_points = torch.cat([single_point.unsqueeze(0), context_points], dim=0)
    
    print(f"Extracting features for point: {single_point[7:10].tolist()}")
    
    # Extract both voxel and relative position features
    out = extractor.extract_features(
        all_points,
        octree_depth=8,
        position_encoding='both'  # displacement + spherical
    )
    
    voxel_features = out['voxel_features']
    relative_positions = out['relative_positions']
    
    # Print results
    print("\nVoxel features (per non-empty voxel):")
    for depth, feature_tensor in voxel_features.items():
        print(f"  Depth {depth}: {feature_tensor.shape}")
    
    print("\nRelative position features (per input point):")
    for depth, rel in relative_positions.items():
        print(f"  Depth {depth}: {rel.shape}  (N x 6 for 'both')")
    
    # Show the target point (index 0) relative position at each depth
    target_idx = 0
    print("\nTarget point relative position (displacement + spherical) per depth:")
    for depth in sorted(relative_positions.keys()):
        rel = relative_positions[depth][target_idx]
        displacement = rel[:3]
        spherical = rel[3:]
        print(f"  Depth {depth}: displacement={displacement.tolist()}, spherical={spherical.tolist()}")
    
    return out


def example_batch_point_feature_extraction():
    """Example of extracting features for multiple points."""
    
    # Configuration
    checkpoint_path = "checkpoints/octformer_scannet200/best_model.pth"  # Use the provided checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create feature extractor
    extractor = FeatureExtractor(
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Create multiple points with NDFP format
    num_points = 10
    points = torch.randn(num_points, 10, dtype=torch.float32)
    points[:, 0:3] = torch.randn(num_points, 3) * 0.1  # Normal
    points[:, 3] = torch.randn(num_points) * 0.1  # Displacement
    points[:, 4:7] = torch.ones(num_points, 3) * 0.5  # Color
    points[:, 7:10] = torch.randn(num_points, 3) * 0.5  # Position
    
    print(f"Extracting features for {num_points} points...")
    
    # Extract both voxel and relative position features
    out = extractor.extract_features(points, octree_depth=8, position_encoding='displacement')
    
    print("\nVoxel features:")
    for depth, feature_tensor in out['voxel_features'].items():
        print(f"  Depth {depth}: {feature_tensor.shape}")
    
    print("\nRelative positions (displacement):")
    for depth, rel in out['relative_positions'].items():
        print(f"  Depth {depth}: {rel.shape}")
    
    return out


def example_save_features():
    """Example of saving extracted features to file."""
    
    # Extract features
    out = example_single_point_feature_extraction()
    
    # Save features to file
    output_path = "extracted_features.pth"
    torch.save(out, output_path)
    print(f"\nFeatures saved to: {output_path}")
    
    # Load features back
    loaded = torch.load(output_path)
    print(f"Features loaded back successfully!")
    
    voxel_levels = len(loaded['voxel_features'])
    rel_levels = len(loaded['relative_positions'])
    print(f"Number of voxel feature levels: {voxel_levels}")
    print(f"Number of relative position levels: {rel_levels}")


if __name__ == "__main__":
    print("OctFormer Feature Extraction Example")
    print("=" * 40)
    
    # Example 1: Single point feature extraction
    print("\n1. Single Point Feature Extraction")
    print("-" * 30)
    try:
        features = example_single_point_feature_extraction()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the checkpoint_path in the script!")
    
    # Example 2: Batch point feature extraction
    print("\n2. Batch Point Feature Extraction")
    print("-" * 30)
    try:
        batch_features = example_batch_point_feature_extraction()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the checkpoint_path in the script!")
    
    # Example 3: Save features
    print("\n3. Save Features")
    print("-" * 30)
    try:
        example_save_features()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the checkpoint_path in the script!")
    
    print("\nDone!") 