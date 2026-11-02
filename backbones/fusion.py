import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLowRankTensorFusion(nn.Module):
    """
    Quantized Low-Rank Tensor Fusion (Q-LRTF) for Multi-Modal Edge Face Recognition.
    
    This module performs bilinear cross-modal fusion factorized via CP-decomposition,
    reducing parameter complexity from O(d^2 h) to O(r(2d + h)).
    It incorporates Fake Quantization and Cross-Modal Scale Bounding (CMSB) regularization.
    """
    def __init__(self, in_features_vis, in_features_nir, out_features, rank=64, quantize=False):
        super(QuantizedLowRankTensorFusion, self).__init__()
        self.in_features_vis = in_features_vis
        self.in_features_nir = in_features_nir
        self.out_features = out_features
        self.rank = rank
        self.quantize = quantize
        
        # Projection matrices: U_1, U_2, U_3 implemented with raw Parameters to avoid quantize_dynamic
        self.U1_weight = nn.Parameter(torch.Tensor(rank, in_features_vis))
        self.U2_weight = nn.Parameter(torch.Tensor(rank, in_features_nir))
        self.U3_weight = nn.Parameter(torch.Tensor(out_features, rank))
        self.U3_bias = nn.Parameter(torch.Tensor(out_features))
        
        self.cmsb_loss = 0.0
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.U1_weight)
        nn.init.xavier_uniform_(self.U2_weight)
        nn.init.xavier_uniform_(self.U3_weight)
        nn.init.zeros_(self.U3_bias)

    def _fake_quantize(self, weight):
        """Simulates symmetric 8-bit quantization with a straight-through estimator"""
        w_max = weight.abs().max()
        scale = w_max / 127.0
        if scale == 0:
            return weight
        quantized = torch.round(weight / scale).clamp(-128, 127)
        # STE: forward uses quantized weights, backward passes gradients through unchanged
        return weight + (quantized * scale - weight).detach()

    def forward(self, x_vis, x_nir):
        """
        x_vis: [batch_size, in_features_vis]
        x_nir: [batch_size, in_features_nir]
        """
        if self.quantize:
            w1 = self._fake_quantize(self.U1_weight)
            w2 = self._fake_quantize(self.U2_weight)
            w3 = self._fake_quantize(self.U3_weight)
            
            u1_out = F.linear(x_vis, w1, None)
            u2_out = F.linear(x_nir, w2, None)
        else:
            u1_out = F.linear(x_vis, self.U1_weight, None)
            u2_out = F.linear(x_nir, self.U2_weight, None)
            
        # Cross-Modal Scale Bounding (CMSB) regularization computation
        if self.training:
            # L2 norms of projections
            norm_u1 = torch.norm(self.U1_weight, p='fro')**2
            norm_u2 = torch.norm(self.U2_weight, p='fro')**2
            norm_u3 = torch.norm(self.U3_weight, p='fro')**2
            
            # Feature magnitude balance
            mag_vis = torch.norm(u1_out, p=2, dim=1)**2
            mag_nir = torch.norm(u2_out, p=2, dim=1)**2
            balance_penalty = torch.abs(mag_vis - mag_nir).mean()
            
            self.cmsb_loss = (norm_u1 + norm_u2 + norm_u3) * 1e-4 + balance_penalty * 1e-3
        else:
            self.cmsb_loss = 0.0

        # Hadamard product
        hadamard_prod = u1_out * u2_out
        
        if self.quantize:
            y = F.linear(hadamard_prod, w3, self.U3_bias)
        else:
            y = F.linear(hadamard_prod, self.U3_weight, self.U3_bias)
            
        return y
