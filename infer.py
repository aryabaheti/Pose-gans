# infer.py

import os
import torch
from utils.visualization import save_comparison_grid

def infer_model(netG, dataloader, device, output_dir="outputs/infer"):
    """
    Run inference using netG on data from dataloader.
    Args:
        netG: generator model (on device)
        dataloader: DataLoader yielding dicts with src_img, tgt_img, src_pose, tgt_pose
        device: torch.device
        output_dir: folder to save output images
    """
    os.makedirs(output_dir, exist_ok=True)
    netG.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src = batch['src_img'].to(device)
            tgt = batch['tgt_img'].to(device)
            sp = batch['src_pose'].to(device)
            tp = batch['tgt_pose'].to(device)

            fake, _ = netG(src, sp, tp)
            out_name = os.path.join(output_dir, f"result_{i}.png")
            save_comparison_grid(src, tgt, tp, fake, out_name, nrow=src.size(0))
    print("Inference done; results saved to", output_dir)
