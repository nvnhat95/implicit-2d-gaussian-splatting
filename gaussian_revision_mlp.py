import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os
import sys

# Add octformer to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'octformer'))

from octformer.gaussian_to_ndfp import GaussianToNDFP
from octformer.extract_features import FeatureExtractor

# Import ocnn for type hints
try:
    import ocnn
except ImportError:
    # ocnn might not be available in all environments, so make it optional for type hints
    ocnn = None


class GaussianDeltaMLP(nn.Module):
    """MLP that takes concatenated octformer features and outputs delta Gaussians."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 128],
                 max_sh_degree: int = 3,
                 dropout: float = 0.1,
                 feature_depth: int = 8):
        super().__init__()
        
        self.max_sh_degree = max_sh_degree
        self.feature_depth = feature_depth
        
        # Calculate output dimensions
        # xyz: 3, features_dc: 3, features_rest: 3 * (max_sh_degree+1)^2 - 3, 
        # scaling: 2, rotation: 4, opacity: 1
        features_rest_dim = 3 * ((max_sh_degree + 1) ** 2 - 1)
        self.output_dim = 3 + 3 + features_rest_dim + 2 + 4 + 1
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer (no activation - we want raw deltas)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning delta parameters.
        
        Args:
            features: (N, input_dim) concatenated features
            
        Returns:
            Dict containing delta parameters for each Gaussian component
        """
        deltas = self.mlp(features)  # (N, output_dim)
        
        # Split into components
        idx = 0
        
        # xyz deltas (3)
        delta_xyz = deltas[:, idx:idx+3]
        idx += 3
        
        # features_dc deltas (3)
        delta_features_dc = deltas[:, idx:idx+3]
        idx += 3
        
        # features_rest deltas
        features_rest_dim = 3 * ((self.max_sh_degree + 1) ** 2 - 1)
        delta_features_rest = deltas[:, idx:idx+features_rest_dim]
        idx += features_rest_dim
        
        # scaling deltas (2)  
        delta_scaling = deltas[:, idx:idx+2]
        idx += 2
        
        # rotation deltas (4)
        delta_rotation = deltas[:, idx:idx+4]
        idx += 4
        
        # opacity deltas (1)
        delta_opacity = deltas[:, idx:idx+1]
        
        return {
            'xyz': delta_xyz,
            'features_dc': delta_features_dc,
            'features_rest': delta_features_rest, 
            'scaling': delta_scaling,
            'rotation': delta_rotation,
            'opacity': delta_opacity
        }


class GaussianRevisionPipeline(nn.Module):
    """Complete pipeline for Gaussian revision using octformer features.
    
    This pipeline includes automatic batching support to handle large numbers of Gaussians
    that may not fit in VRAM when processed all at once. The batching mechanism:
    
    1. Splits large point sets into smaller batches during feature extraction and MLP forward pass
    2. Processes each batch independently to reduce peak memory usage
    3. Concatenates results to produce the same output as non-batched processing
    4. Includes automatic memory cleanup between batches
    
    Usage:
        # Default batch size (8192) - good for most GPUs
        pipeline = GaussianRevisionPipeline(checkpoint_path)
        deltas = pipeline(gaussians)
        
        # Auto-detect optimal batch size based on GPU memory
        recommended_batch_size = GaussianRevisionPipeline.recommend_batch_size()
        pipeline = GaussianRevisionPipeline(checkpoint_path, batch_size=recommended_batch_size)
        
        # Custom batch size for smaller/larger VRAM
        pipeline = GaussianRevisionPipeline(checkpoint_path, batch_size=4096)  # Smaller for low VRAM
        deltas = pipeline(gaussians)
        
        # Disable batching (process all at once)
        pipeline = GaussianRevisionPipeline(checkpoint_path, batch_size=None)
        deltas = pipeline(gaussians)
        
        # Override batch size per call
        deltas = pipeline(gaussians, batch_size=2048)
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 max_sh_degree: int = 3,
                 octree_depth: int = 8,
                 position_encoding: str = 'both',
                 feature_depth: int = 8,
                 mlp_hidden_dims: list = [128, 128],
                 mlp_dropout: float = 0.1,
                 batch_size: Optional[int] = 8192,
                 device: str = 'cuda'):
        super().__init__()
        
        self.octree_depth = octree_depth
        self.position_encoding = position_encoding
        self.feature_depth = feature_depth
        self.batch_size = batch_size
        self.device = device
        
        # Initialize NDFP converter
        self.ndfp_converter = GaussianToNDFP(device=device)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(checkpoint_path, device)
        
        # Get feature dimensions by running a dummy forward pass
        with torch.no_grad():
            dummy_ndfp = torch.randn(100, 10, device=device)
            dummy_features = self.feature_extractor.extract_features(
                dummy_ndfp, octree_depth, position_encoding
            )
            input_dim = self._calculate_input_dim(dummy_features)
        
        # Initialize MLP and move to device
        self.mlp = GaussianDeltaMLP(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dims,
            max_sh_degree=max_sh_degree,
            dropout=mlp_dropout,
            feature_depth=feature_depth
        ).to(device)
        
        # Move the entire pipeline to the specified device
        self.to(device)
        
        print(f"Initialized GaussianRevisionPipeline with input_dim={input_dim}, batch_size={batch_size}")
    
    @staticmethod
    def recommend_batch_size() -> int:
        """Recommend an appropriate batch size based on available GPU memory.
        
        Returns:
            Recommended batch size for current GPU
        """
        if not torch.cuda.is_available():
            return 2048  # Conservative for CPU
            
        try:
            # Get GPU memory info
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            available_memory = total_memory - allocated_memory
            
            # Convert to GB
            available_gb = available_memory / (1024**3)
            
            # Conservative recommendations based on memory
            if available_gb >= 20:
                return 16384  # Large batch for high-end GPUs
            elif available_gb >= 12:
                return 8192   # Default for mid-range GPUs
            elif available_gb >= 8:
                return 4096   # Smaller for 8GB GPUs
            elif available_gb >= 4:
                return 2048   # Very conservative for 4GB GPUs
            else:
                return 1024   # Minimal for low memory
                
        except Exception:
            return 4096  # Safe default if memory query fails
    
    def _calculate_input_dim(self, features_dict: Dict) -> int:
        """Calculate the total input dimension for the MLP."""
        voxel_features = features_dict['voxel_features']
        relative_positions = features_dict['relative_positions']
        
        # Use features from specific depth only
        if self.feature_depth in voxel_features:
            voxel_dim = voxel_features[self.feature_depth].shape[1]
        else:
            # Use the deepest available if specified depth doesn't exist
            voxel_dim = voxel_features[max(voxel_features.keys())].shape[1]
        
        # Relative position dimension from the same depth
        if self.feature_depth in relative_positions:
            rel_pos_dim = relative_positions[self.feature_depth].shape[1]
        else:
            # Use the deepest available if specified depth doesn't exist
            rel_pos_dim = relative_positions[max(relative_positions.keys())].shape[1]
        
        return voxel_dim + rel_pos_dim
    
    def _get_point_to_voxel_mapping(self, ndfp_points: torch.Tensor, octree, depth: int, batch_size: int = 10000, use_cpu_fallback: bool = True) -> torch.Tensor:
        """Get mapping from points to voxel indices at given depth with memory management.
        
        Args:
            ndfp_points: Tensor (N, 10) in NDFP format
            octree: Built octree
            depth: Octree depth to use
            batch_size: Number of points to process at once to avoid OOM
            use_cpu_fallback: If True, fallback to CPU computation on OOM
            
        Returns:
            Tensor (N,) mapping each point to its voxel index
        """
        point_positions = ndfp_points[:, 7:10]  # Extract XYZ coordinates
        
        # Get voxel keys for this depth
        try:
            keys = octree.key(depth, True)  # Get non-empty voxel keys
        except Exception:
            keys = octree.key(depth)
        
        if len(keys) == 0:
            # No voxels at this depth, return zeros
            return torch.zeros(point_positions.shape[0], dtype=torch.long, device=point_positions.device)
        
        # Convert keys to voxel centers
        x, y, z, _ = ocnn.octree.key2xyz(keys, depth)
        voxel_centers = torch.stack([x, y, z], dim=1).to(torch.float32)
        
        # Memory-efficient closest voxel assignment
        torch.cuda.empty_cache()  # Clear GPU cache
        
        try:
            # Try batched computation on GPU
            closest_voxel = self._compute_closest_voxels_batched(point_positions, voxel_centers, batch_size)
        except RuntimeError as e2:
            if ("out of memory" in str(e2).lower() or "cuda out of memory" in str(e2).lower()) and use_cpu_fallback:
                print(f"GPU batched OOM at depth {depth}. Falling back to CPU computation...")
                torch.cuda.empty_cache()
                # Move to CPU for computation
                closest_voxel = self._compute_closest_voxels_cpu(point_positions, voxel_centers, batch_size)
            else:
                raise e2
        
        return closest_voxel

    def _compute_closest_voxels(self, point_positions: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute closest voxels using full distance matrix (fastest but most memory intensive)."""
        distances = torch.cdist(point_positions, centers)
        return torch.argmin(distances, dim=1)

    def _compute_closest_voxels_batched(self, point_positions: torch.Tensor, centers: torch.Tensor, batch_size: int = 10000) -> torch.Tensor:
        """Compute closest voxels using batched approach to reduce memory usage."""
        num_points = point_positions.shape[0]
        closest_voxels = torch.empty(num_points, dtype=torch.long, device=point_positions.device)
        
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_points = point_positions[start_idx:end_idx]
            
            # Compute distances for this batch
            batch_distances = torch.cdist(batch_points, centers)
            batch_closest = torch.argmin(batch_distances, dim=1)
            
            closest_voxels[start_idx:end_idx] = batch_closest
            
        return closest_voxels

    def _compute_closest_voxels_cpu(self, point_positions: torch.Tensor, centers: torch.Tensor, batch_size: int = 10000) -> torch.Tensor:
        """Compute closest voxels on CPU with batching (slowest but most memory efficient)."""
        # Move to CPU
        point_positions_cpu = point_positions.cpu()
        centers_cpu = centers.cpu()
        
        num_points = point_positions_cpu.shape[0]
        closest_voxels = torch.empty(num_points, dtype=torch.long)
        
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_points = point_positions_cpu[start_idx:end_idx]
            
            # Compute distances for this batch on CPU
            batch_distances = torch.cdist(batch_points, centers_cpu)
            batch_closest = torch.argmin(batch_distances, dim=1)
            
            closest_voxels[start_idx:end_idx] = batch_closest
            
        # Move result back to original device
        return closest_voxels.to(point_positions.device)

    def _aggregate_features(self, 
                          voxel_features: Dict[int, torch.Tensor],
                          relative_positions: Dict[int, torch.Tensor],
                          num_points: int,
                          ndfp_points: torch.Tensor = None,
                          octree = None) -> torch.Tensor:
        """Extract voxel and relative position features from specific depth.
        
        Args:
            voxel_features: Dict mapping depth to voxel features
            relative_positions: Dict mapping depth to relative position features  
            num_points: Number of points
            ndfp_points: NDFP points for mapping (required if voxel mapping needed)
            octree: Octree for mapping (required if voxel mapping needed)
        """
        
        # Use features from specific depth only
        depth = self.feature_depth if self.feature_depth in voxel_features else max(voxel_features.keys())
        
        # Get voxel features and relative positions for this depth
        all_voxel_features = voxel_features[depth]  # (num_voxels, feat_dim)
        rel_pos_feat = relative_positions[depth]    # (num_points, rel_pos_dim)
        
        # Check if we have the right number of features for our points
        if all_voxel_features.shape[0] != num_points:
            if ndfp_points is not None and octree is not None:
                # Proper mapping: get which voxel each point belongs to
                point_to_voxel = self._get_point_to_voxel_mapping(ndfp_points, octree, depth, batch_size=10000, use_cpu_fallback=True)
                # Index voxel features using the mapping
                point_voxel_features = all_voxel_features[point_to_voxel]
            else:
                # Fallback: use first num_points features (not ideal but prevents crash)
                print(f"Warning: voxel features shape {all_voxel_features.shape} doesn't match num_points {num_points}")
                print("Using first num_points voxel features as fallback (ndfp_points or octree not provided)")
                if all_voxel_features.shape[0] >= num_points:
                    point_voxel_features = all_voxel_features[:num_points]
                else:
                    # If we have fewer voxel features than points, repeat the last feature
                    point_voxel_features = torch.cat([
                        all_voxel_features,
                        all_voxel_features[-1:].repeat(num_points - all_voxel_features.shape[0], 1)
                    ], dim=0)
        else:
            point_voxel_features = all_voxel_features
        
        # Ensure all features are on the same device before concatenation
        point_voxel_features = point_voxel_features.to(self.device)
        rel_pos_feat = rel_pos_feat.to(self.device)
        
        # Concatenate voxel and relative position features
        combined_features = torch.cat([point_voxel_features, rel_pos_feat], dim=1)
        return combined_features
    
    def _forward_single_batch(self, batch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single batch of concatenated features through the MLP.
        
        Args:
            batch_features: Tensor (N, input_dim) concatenated features for the batch
            
        Returns:
            Dict containing delta parameters
        """
        # Ensure features are on the correct device
        batch_features = batch_features.to(self.device)
        
        # Predict deltas (this needs gradients during training)
        deltas = self.mlp(batch_features)
        return deltas
    
    def _forward_batched(self, combined_features: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Process concatenated features in batches and concatenate results.
        
        Args:
            combined_features: Tensor (N, input_dim) concatenated features for all points
            batch_size: Size of each batch
            
        Returns:
            Dict containing concatenated delta parameters
        """
        n_points = combined_features.shape[0]
        n_batches = (n_points + batch_size - 1) // batch_size
        all_deltas = {}
        
        print(f"Processing {n_points} points in {n_batches} batches...")
        
        # Process in batches
        for batch_idx, start_idx in enumerate(range(0, n_points, batch_size)):
            end_idx = min(start_idx + batch_size, n_points)
            batch_features = combined_features[start_idx:end_idx]
            
            # Process this batch through MLP
            batch_deltas = self._forward_single_batch(batch_features)
            
            # Initialize output tensors on first batch
            if start_idx == 0:
                for key, value in batch_deltas.items():
                    all_deltas[key] = torch.empty(
                        (n_points, value.shape[1]), 
                        dtype=value.dtype, 
                        device=value.device
                    )
            
            # Store batch results
            for key, value in batch_deltas.items():
                all_deltas[key][start_idx:end_idx] = value
            
            # Clear intermediate tensors to free memory
            del batch_deltas
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("Batch processing completed.")
        return all_deltas

    def forward(self, gaussians, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass: convert gaussians to NDFP, extract features, predict deltas.
        
        Args:
            gaussians: GaussianModel instance
            batch_size: Optional batch size override. If None, uses self.batch_size
            
        Returns:
            Dict containing delta parameters
        """
        # Convert to NDFP format
        ndfp_points = self.ndfp_converter.convert_from_model(gaussians)
        
        # Build octree once with all points
        print(f"Building octree from {ndfp_points.shape[0]} points...")
        octree = self.feature_extractor.build_octree(ndfp_points, self.octree_depth)
        print("Octree built successfully.")
        
        # Extract features once for all points
        print(f"Extracting features for {ndfp_points.shape[0]} points...")
        if self.training:
            # During training, we need gradients for MLP, but feature extraction can be no_grad
            with torch.no_grad():
                features_dict = self.feature_extractor.extract_features_from_octree(
                    ndfp_points, octree, self.position_encoding, batch_size=batch_size, use_cpu_fallback=True
                )
            
            # Aggregate features with gradients enabled for MLP
            combined_features = self._aggregate_features(
                features_dict['voxel_features'],
                features_dict['relative_positions'],
                ndfp_points.shape[0],
                ndfp_points,
                octree
            )
        else:
            # During inference, everything can be no_grad
            with torch.no_grad():
                features_dict = self.feature_extractor.extract_features_from_octree(
                    ndfp_points, octree, self.position_encoding, batch_size=batch_size, use_cpu_fallback=True
                )
                combined_features = self._aggregate_features(
                    features_dict['voxel_features'],
                    features_dict['relative_positions'],
                    ndfp_points.shape[0],
                    ndfp_points,
                    octree
                )
        
        print("Features extracted successfully.")
        
        # Determine effective batch size
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        
        # If batch_size is None or features fit in memory, process all at once
        if effective_batch_size is None or combined_features.shape[0] <= effective_batch_size:
            print(f"Processing {combined_features.shape[0]} points in single batch")
            return self._forward_single_batch(combined_features)
        
        # Otherwise, process in batches
        print(f"Processing {combined_features.shape[0]} points in batches of {effective_batch_size}")
        return self._forward_batched(combined_features, effective_batch_size)


def apply_gaussian_deltas(gaussians, deltas: Dict[str, torch.Tensor], alpha: float = 0.5, training: bool = True):
    """Apply delta parameters to gaussians with blending factor alpha.
    
    Args:
        gaussians: GaussianModel instance
        deltas: Dict containing delta parameters
        alpha: Blending factor (0 = no change, 1 = full delta)
        training: Whether we're in training mode (affects gradient handling)
    """
    if training:
        # During training, modify the parameters while preserving gradients
        
        # XYZ - in-place operations preserve gradients
        gaussians._xyz.data.add_(deltas['xyz'], alpha=alpha)
        
        # Features DC - need to handle the shape (N, 1, 3)
        delta_dc = deltas['features_dc'].unsqueeze(1)  # (N, 1, 3)
        gaussians._features_dc.data.add_(delta_dc, alpha=alpha)
        
        # Features Rest - need to reshape properly
        if gaussians._features_rest.numel() > 0:
            n_points = gaussians._features_rest.shape[0]
            n_features = gaussians._features_rest.shape[1] 
            n_coeffs = gaussians._features_rest.shape[2]
            
            # Reshape delta to match features_rest shape
            delta_rest = deltas['features_rest'].view(n_points, n_features, n_coeffs)
            gaussians._features_rest.data.add_(delta_rest, alpha=alpha)
        
        # Scaling
        gaussians._scaling.data.add_(deltas['scaling'], alpha=alpha)
        
        # Rotation
        gaussians._rotation.data.add_(deltas['rotation'], alpha=alpha)
        
        # Opacity  
        gaussians._opacity.data.add_(deltas['opacity'], alpha=alpha)
        
    else:
        # During inference, use no_grad for efficiency
        with torch.no_grad():
            # Apply deltas to each parameter
            
            # XYZ
            gaussians._xyz.data = gaussians._xyz.data + alpha * deltas['xyz']
            
            # Features DC - need to handle the shape (N, 1, 3)
            delta_dc = deltas['features_dc'].unsqueeze(1)  # (N, 1, 3)
            gaussians._features_dc.data = gaussians._features_dc.data + alpha * delta_dc
            
            # Features Rest - need to reshape properly
            if gaussians._features_rest.numel() > 0:
                n_points = gaussians._features_rest.shape[0]
                n_features = gaussians._features_rest.shape[1] 
                n_coeffs = gaussians._features_rest.shape[2]
                
                # Reshape delta to match features_rest shape
                delta_rest = deltas['features_rest'].view(n_points, n_features, n_coeffs)
                gaussians._features_rest.data = gaussians._features_rest.data + alpha * delta_rest
            
            # Scaling
            gaussians._scaling.data = gaussians._scaling.data + alpha * deltas['scaling']
            
            # Rotation
            gaussians._rotation.data = gaussians._rotation.data + alpha * deltas['rotation']
            
            # Opacity  
            gaussians._opacity.data = gaussians._opacity.data + alpha * deltas['opacity'] 