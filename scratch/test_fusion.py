import torch
import sys
import os

# Add parent directory to path so we can import backbones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbones import get_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_mflops(model, x_vis, x_nir):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True
    ) as prof:
        _ = model(x_vis, x_nir)
        
    total_flops = sum(record.flops for record in prof.key_averages() if hasattr(record, 'flops'))
    return total_flops / 1e6
import traceback

if __name__ == '__main__':
    print("Testing Dual-Stream Fusion Architectures...\n")
    
    # Dummy VIS and NIR inputs
    x_vis = torch.rand(2, 3, 112, 112)
    x_nir = torch.rand(2, 3, 112, 112)
    
    models_to_test = [
        'edgeface_fusion_xs',
        'edgeface_fusion_xs_gamma_06',
        'edgeface_fusion_xs_gamma_06_q'
    ]
    
    for m_name in models_to_test:
        print(f"--- Testing {m_name} ---")
        model = get_model(m_name)
        model.eval()
        
        # Test forward pass
        try:
            out = model(x_vis, x_nir)
            print(f"Forward Pass Success! Output shape: {out.shape}")
        except Exception as e:
            print(f"Forward Pass FAILED: {e}")
            traceback.print_exc()
            continue
            
        # Get parameter counts
        params = count_parameters(model) / 1e6
        print(f"Parameters: {params:.3f} M")
        
        # Get MFLOPs
        flops = get_mflops(model, x_vis, x_nir)
        print(f"MFLOPs: {flops:.3f}")
        
        print(f"CMSB Loss: {model.fusion.cmsb_loss}\n")
