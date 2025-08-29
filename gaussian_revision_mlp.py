import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os
import sys
import copy

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
    """MLP that takes Gaussian parameters as input and outputs delta Gaussians."""
    
    def __init__(self, 
                 max_sh_degree: int = 3,
                 hidden_dims: list = [128, 128],
                 dropout: float = 0.0):
        super().__init__()
        
        self.max_sh_degree = max_sh_degree
        
        # Calculate input dimensions
        # xyz: 3, features_dc: 3, features_rest: 3 * (max_sh_degree+1)^2 - 3, 
        # scaling: 2, rotation: 4, opacity: 1
        features_rest_dim = 3 * ((max_sh_degree + 1) ** 2 - 1)
        self.input_dim = 3 + 3 + features_rest_dim + 2 + 4 + 1
        
        # Calculate output dimensions (same as input for delta prediction)
        self.output_dim = self.input_dim
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Sigmoid(),
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
    
    def _flatten_gaussian_params(self, gaussian_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten Gaussian parameters into a single tensor for MLP input.
        
        Args:
            gaussian_params: Dict containing Gaussian parameters
            
        Returns:
            Flattened tensor (N, input_dim)
        """
        # Extract and flatten each parameter
        xyz = gaussian_params['xyz']  # (N, 3)
        
        # Handle features_dc which might be (N, 1, 3)
        features_dc = gaussian_params['features_dc']
        if features_dc.dim() == 3:
            features_dc = features_dc.squeeze(1)  # (N, 3)
        
        # Handle features_rest which might be (N, C, 3)
        features_rest = gaussian_params['features_rest']
        if features_rest.dim() == 3:
            N, C, D = features_rest.shape
            features_rest = features_rest.reshape(N, C * D)  # (N, C*3)
        
        scaling = gaussian_params['scaling']  # (N, 2)
        rotation = gaussian_params['rotation']  # (N, 4)
        opacity = gaussian_params['opacity']  # (N, 1)
        
        # Concatenate all parameters
        flattened = torch.cat([
            xyz, 
            features_dc, 
            features_rest, 
            scaling, 
            rotation, 
            opacity
        ], dim=1)
        
        return flattened
    
    def _unflatten_deltas(self, deltas: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split flattened deltas back into parameter components.
        
        Args:
            deltas: (N, output_dim) tensor of deltas
            
        Returns:
            Dict containing delta parameters for each Gaussian component
        """
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
    
    def forward(self, gaussian_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass returning delta parameters.
        
        Args:
            gaussian_params: Dict containing Gaussian parameters
            
        Returns:
            Dict containing delta parameters for each Gaussian component
        """
        # Flatten Gaussian parameters for MLP input
        flattened_params = self._flatten_gaussian_params(gaussian_params)
        
        # Run through MLP
        flattened_deltas = self.mlp(flattened_params)
        
        # Unflatten deltas back to parameter dict
        return self._unflatten_deltas(flattened_deltas)


class GaussianRevisionPipeline(nn.Module):
    """Complete pipeline for Gaussian revision using direct Gaussian parameters.
    
    This pipeline includes automatic batching support to handle large numbers of Gaussians
    that may not fit in VRAM when processed all at once. The batching mechanism:
    
    1. Splits large point sets into smaller batches during MLP forward pass
    2. Processes each batch independently to reduce peak memory usage
    3. Concatenates results to produce the same output as non-batched processing
    4. Includes automatic memory cleanup between batches
    
    Usage:
        # Default batch size (8192) - good for most GPUs
        pipeline = GaussianRevisionPipeline(max_sh_degree=3)
        deltas = pipeline(gaussians)
        
        # Auto-detect optimal batch size based on GPU memory
        recommended_batch_size = GaussianRevisionPipeline.recommend_batch_size()
        pipeline = GaussianRevisionPipeline(batch_size=recommended_batch_size)
        
        # Custom batch size for smaller/larger VRAM
        pipeline = GaussianRevisionPipeline(batch_size=4096)  # Smaller for low VRAM
        deltas = pipeline(gaussians)
        
        # Disable batching (process all at once)
        pipeline = GaussianRevisionPipeline(batch_size=None)
        deltas = pipeline(gaussians)
        
        # Override batch size per call
        deltas = pipeline(gaussians, batch_size=2048)
    """
    
    def __init__(self,
                 max_sh_degree: int = 3,
                 mlp_hidden_dims: list = [128, 128],
                 mlp_dropout: float = 0.1,
                 batch_size: Optional[int] = 8192,
                 device: str = 'cuda'):
        super().__init__()
        
        self.max_sh_degree = max_sh_degree
        self.batch_size = batch_size
        self.device = device
        
        # Initialize MLP and move to device
        self.mlp = GaussianDeltaMLP(
            max_sh_degree=max_sh_degree,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout
        ).to(device)
        
        # Move the entire pipeline to the specified device
        self.to(device)
        
        print(f"Initialized GaussianRevisionPipeline with batch_size={batch_size}")
    
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
    
    def _extract_gaussian_params(self, gaussians) -> Dict[str, torch.Tensor]:
        """Extract parameters from Gaussian model into a dictionary.
        
        Args:
            gaussians: GaussianModel instance
            
        Returns:
            Dict containing Gaussian parameters
        """
        return {
            'xyz': gaussians._xyz,
            'features_dc': gaussians._features_dc,
            'features_rest': gaussians._features_rest,
            'scaling': gaussians._scaling,
            'rotation': gaussians._rotation,
            'opacity': gaussians._opacity
        }
    
    def _forward_single_batch(self, batch_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a single batch of Gaussian parameters through the MLP.
        
        Args:
            batch_params: Dict containing Gaussian parameters for the batch
            
        Returns:
            Dict containing delta parameters
        """
        # Ensure parameters are on the correct device
        for key in batch_params:
            batch_params[key] = batch_params[key].to(self.device)
        
        # Predict deltas (this needs gradients during training)
        deltas = self.mlp(batch_params)
        return deltas
    
    def _forward_batched(self, gaussian_params: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process Gaussian parameters in batches and concatenate results.
        
        Args:
            gaussian_params: Dict containing Gaussian parameters for all points
            batch_size: Size of each batch
            
        Returns:
            Dict containing concatenated delta parameters
        """
        n_points = gaussian_params['xyz'].shape[0]
        n_batches = (n_points + batch_size - 1) // batch_size
        batch_results = []
        
        # Process in batches
        for batch_idx, start_idx in enumerate(range(0, n_points, batch_size)):
            end_idx = min(start_idx + batch_size, n_points)
            
            # Create batch parameters
            batch_params = {}
            for key, value in gaussian_params.items():
                batch_params[key] = value[start_idx:end_idx]
            
            # Process this batch through MLP
            batch_deltas = self._forward_single_batch(batch_params)
            batch_results.append(batch_deltas)
            
            # Clear intermediate tensors to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batch results to preserve gradients
        all_deltas = {}
        for key in batch_results[0].keys():
            all_deltas[key] = torch.cat([batch[key] for batch in batch_results], dim=0)
        
        return all_deltas

    def forward(self, gaussians, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass: extract Gaussian parameters and predict deltas.
        
        Args:
            gaussians: GaussianModel instance
            batch_size: Optional batch size override. If None, uses self.batch_size
            
        Returns:
            Dict containing delta parameters
        """
        # Extract Gaussian parameters
        gaussian_params = self._extract_gaussian_params(gaussians)
        
        # Determine effective batch size
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        
        # If batch_size is None or parameters fit in memory, process all at once
        if effective_batch_size is None or gaussian_params['xyz'].shape[0] <= effective_batch_size:
            return self._forward_single_batch(gaussian_params)
        
        # Otherwise, process in batches
        return self._forward_batched(gaussian_params, effective_batch_size)


def apply_gaussian_deltas(
    base_gaussians,
    deltas: Dict[str, torch.Tensor], 
    alpha: float = 0.5, 
    training: bool = True
) -> 'GaussianModel':
    """Apply delta parameters to base gaussians and return a modified copy.
    
    Args:
        base_gaussians: Original unmodified GaussianModel instance
        deltas: Dict containing delta parameters
        alpha: Blending factor (0 = no change, 1 = full delta)
        training: Whether we're in training mode (affects gradient handling)
        
    Returns:
        Modified copy of the gaussians with deltas applied
    """
    # Create a copy of base gaussians for modification
    modified_gaussians = copy.deepcopy(base_gaussians)
    
    if training:
        # Apply deltas with gradient flow for MLP learning
        # Compute terms separately for debugging

        # print("XYZ compared", (1 - alpha) * torch.mean(torch.abs(base_gaussians._xyz)), alpha * torch.mean(torch.abs(deltas['xyz'])))
        # print("features_dc compared", (1 - alpha) * torch.mean(torch.abs(base_gaussians._features_dc)), alpha * torch.mean(torch.abs(deltas['features_dc'])))
        # print("features_rest compared", (1 - alpha) * torch.mean(torch.abs(base_gaussians._features_rest)), alpha * torch.mean(torch.abs(deltas['features_rest'])))
        # print("scaling compared", (1 - alpha) * torch.mean(torch.abs(base_gaussians._scaling)), alpha * torch.mean(torch.abs(deltas['scaling'])))
        # print("rotation compared", (1 - alpha) * torch.mean(torch.abs(base_gaussians._rotation)), alpha * torch.mean(torch.abs(deltas['rotation'])))
        # print("opacity compared", (1 - alpha) * torch.mean(torch.abs(base_gaussians._opacity)), alpha * torch.mean(torch.abs(deltas['opacity'])))
        
        # XYZ - blend base with deltas
        modified_gaussians._xyz = (1 - alpha) * base_gaussians._xyz + alpha * deltas['xyz']

        # Features DC - need to handle the shape (N, 1, 3)
        delta_dc = deltas['features_dc'].unsqueeze(1)  # (N, 1, 3)
        modified_gaussians._features_dc = (1 - alpha) * base_gaussians._features_dc + alpha * delta_dc
        
        # Features Rest - need to reshape properly
        if base_gaussians._features_rest.numel() > 0:
            n_points = base_gaussians._features_rest.shape[0]
            n_features = base_gaussians._features_rest.shape[1] 
            n_coeffs = base_gaussians._features_rest.shape[2]
            
            # Reshape delta to match features_rest shape
            delta_rest = deltas['features_rest'].view(n_points, n_features, n_coeffs)
            modified_gaussians._features_rest = (1 - alpha) * base_gaussians._features_rest + alpha * delta_rest
        
        # Scaling
        modified_gaussians._scaling = (1 - alpha) * base_gaussians._scaling + alpha * deltas['scaling']
        
        # Rotation
        modified_gaussians._rotation = (1 - alpha) * base_gaussians._rotation + alpha * deltas['rotation']
        
        # Opacity  
        modified_gaussians._opacity = (1 - alpha) * base_gaussians._opacity + alpha * deltas['opacity']
        
    else:
        # During inference, use no_grad for efficiency
        with torch.no_grad():
            # Apply deltas to each parameter
            
            # XYZ
            modified_gaussians._xyz.data = (1 - alpha) * base_gaussians._xyz.data + alpha * deltas['xyz']
            
            # Features DC - need to handle the shape (N, 1, 3)
            delta_dc = deltas['features_dc'].unsqueeze(1)  # (N, 1, 3)
            modified_gaussians._features_dc.data = (1 - alpha) * base_gaussians._features_dc.data + alpha * delta_dc
            
            # Features Rest - need to reshape properly
            if base_gaussians._features_rest.numel() > 0:
                n_points = base_gaussians._features_rest.shape[0]
                n_features = base_gaussians._features_rest.shape[1] 
                n_coeffs = base_gaussians._features_rest.shape[2]
                
                # Reshape delta to match features_rest shape
                delta_rest = deltas['features_rest'].view(n_points, n_features, n_coeffs)
                modified_gaussians._features_rest.data = (1 - alpha) * base_gaussians._features_rest.data + alpha * delta_rest
            
            # Scaling
            modified_gaussians._scaling.data = (1 - alpha) * base_gaussians._scaling.data + alpha * deltas['scaling']
            
            # Rotation
            modified_gaussians._rotation.data = (1 - alpha) * base_gaussians._rotation.data + alpha * deltas['rotation']
            
            # Opacity  
            modified_gaussians._opacity.data = (1 - alpha) * base_gaussians._opacity.data + alpha * deltas['opacity']
    
    return modified_gaussians 