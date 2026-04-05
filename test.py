import os
import random
import argparse
import time
import cv2
from datetime import datetime
import numpy as np
import json
import torch
from options import TestOptions
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import map_coordinates

import tracetrans

args = TestOptions().parse_args()

with open(args.pairlist_path, "r") as f:
    flist = json.load(f)
dataset_root_path = os.path.dirname(args.pairlist_path)
inshape = [256, 256]

def load_image(file_path, use_rgb=False):
    """Helper function to load and process a single image"""
    img = cv2.imread(os.path.join(dataset_root_path, file_path))
    if use_rgb:
        img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, dsize=tuple(inshape), interpolation=cv2.INTER_LINEAR)
    
    if not use_rgb:
        img = img[..., None]
    
    return (img.astype(np.float32) / 127.5 - 1.0)[None, ...]

torch.backends.cudnn.deterministic = not args.cudnn_nondet

model = tracetrans.tracetrans_model.TraceTransModel(args, is_train=False)
model.load(args.ckpt_path)

output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
subdirs = ("source", "target", "gen", "warped", "gt_warped", "deform_grid", "deform_heatmap")
for subdir in subdirs:
    os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

for i in tqdm(range(len(flist['dataset']))):
    file = flist['dataset'][i]
    img_index = file['idx']
    
    target_img = load_image(file["after"], args.use_rgb)
    source_img = load_image(file["before"], args.use_rgb)
    gt_warped_img = load_image(file["gt_warped"], args.use_rgb)

    source_img = torch.from_numpy(source_img).permute((0, 3, 1, 2)) # 1*H*W*C -> 1*C*H*W
    target_img = torch.from_numpy(target_img).permute((0, 3, 1, 2))
    gt_warped_img = torch.from_numpy(gt_warped_img).permute((0, 3, 1, 2))
    model.set_input(source_img, target_img, gt_warped_img)
    gen_y, warp_result = model.predict()
    warped_x, pos_field = warp_result
    
    tmp_warped_x = ((warped_x[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    tmp_gen_y = ((gen_y[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    tmp_target_img = ((target_img[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    tmp_source_img = ((source_img[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    tmp_gtwarped_img = ((gt_warped_img[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)

    cv2.imwrite(os.path.join(output_path, "warped", f"{img_index}_warped_x.png"), tmp_warped_x)
    cv2.imwrite(os.path.join(output_path, "gen", f"{img_index}_gen_y.png"), tmp_gen_y)
    cv2.imwrite(os.path.join(output_path, "target", f"{img_index}_target.png"), tmp_target_img)
    cv2.imwrite(os.path.join(output_path, "source", f"{img_index}_source.png"), tmp_source_img)
    cv2.imwrite(os.path.join(output_path, "gt_warped", f"{img_index}_gtwarped.png"), tmp_gtwarped_img)
    
    def plot_deformation_grid(pos_field, out_path, step=16):
        pos_field = pos_field[0]  # first batch
        pos_field = pos_field.detach().cpu().numpy()  # [2, H, W]
        H, W = pos_field.shape[1:]

        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        new_y = grid_y + pos_field[0]
        new_x = grid_x + pos_field[1]
        
        plt.figure(figsize=(8, 8))
        for i in range(0, H, step):
            plt.plot(new_x[i, ::step], new_y[i, ::step], 'b-')
        for j in range(0, W, step):
            plt.plot(new_x[::step, j], new_y[::step, j], 'b-')
        plt.gca().invert_yaxis()
        plt.title("Deformation Field Grid (moving→fixed)")
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        
        inverse_field = np.zeros_like(pos_field)
        inverse_field[0] = map_coordinates(-pos_field[0], [new_y, new_x], order=1, mode='nearest')
        inverse_field[1] = map_coordinates(-pos_field[1], [new_y, new_x], order=1, mode='nearest')
        
        inv_new_y = grid_y + inverse_field[0]
        inv_new_x = grid_x + inverse_field[1]
        plt.figure(figsize=(8, 8))
        for i in range(0, H, step):
            plt.plot(inv_new_x[i, ::step], inv_new_y[i, ::step], 'r-')
        for j in range(0, W, step):
            plt.plot(inv_new_x[::step, j], inv_new_y[::step, j], 'r-')
        plt.gca().invert_yaxis()
        plt.title("Inverse Deformation Field Grid (fixed→moving)")
        plt.axis('off')
        plt.savefig(out_path.replace('.png', '_inverse.png'), bbox_inches='tight')
        plt.close()

    plot_deformation_grid(pos_field, os.path.join(output_path, "deform_grid", f"{img_index}_deform_grid.png"))

    # deformatioin heatmap
    def plot_deformation_heatmap(pos_field, out_path):
        pos_field = pos_field[0].detach().cpu().numpy()  # [2, H, W]
        magnitude = np.sqrt(np.sum(pos_field**2, axis=0))  # [H, W]
        plt.figure(figsize=(8, 8))
        plt.imshow(magnitude, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Deformation Magnitude')
        plt.title('Deformation Field Magnitude Heatmap')
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    plot_deformation_heatmap(pos_field, os.path.join(output_path, "deform_heatmap", f"{img_index}_deform_heatmap.png"))