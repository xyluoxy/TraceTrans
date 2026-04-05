import os
import random
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import cv2
from options import TrainOptions

import tracetrans

from utils import *

args = TrainOptions().parse_args()

pair_loader = tracetrans.data_loaders.img_to_img(
    args.pairlist_path, 
    batch_size=args.batch_size,
    output_shape=[args.img_size, args.img_size],
    use_rgb=args.use_rgb,
    use_depth=args.use_depth
)

# extract shape from sampled input
inshape = next(pair_loader)[-1].shape[1:-1]

os.makedirs(args.model_dir, exist_ok=True)

torch.backends.cudnn.deterministic = not args.cudnn_nondet

model = tracetrans.tracetrans_model.TraceTransModel(args, True)
if args.ckpt_path != None:
    print(f"checkpoint loaded from {args.ckpt_path}")
    model.load(args.ckpt_path)
    
# validation pair
val_img_indices, val_source_images, val_target_images, val_gtwarped_images = next(pair_loader)
val_img_index = val_img_indices[0]
val_source_img = val_source_images[0]
val_target_img = val_target_images[0]
val_gtwarped_img = val_gtwarped_images[0]
val_source_img = torch.from_numpy(val_source_img).unsqueeze(0).permute((0, 3, 1, 2))
val_target_img = torch.from_numpy(val_target_img).unsqueeze(0).permute((0, 3, 1, 2))
val_gtwarped_img = torch.from_numpy(val_gtwarped_img).unsqueeze(0).permute((0, 3, 1, 2))

global_step = 0
total_step = args.epochs * args.steps_per_epoch
adv_warm_up = 0
gen_proportion = args.gen_proportion

for epoch in range(args.initial_epoch, args.epochs):
    for step in range(args.steps_per_epoch):
        _, source_img, target_img, gt_warped_img = next(pair_loader)
        source_img = torch.from_numpy(source_img).permute((0, 3, 1, 2)) # B*H*W*C -> B*C*H*W
        target_img = torch.from_numpy(target_img).permute((0, 3, 1, 2))
        gt_warped_img = torch.from_numpy(gt_warped_img).permute((0, 3, 1, 2))
        model.set_input(source_img, target_img, gt_warped_img)
        
        loss_dict_G, loss_dict_D = model.optimize_parameters(lambda_adv=0.01, gen_proportion=gen_proportion)
            
        g_str = ', '.join([f"G_{k}: {v:.4f}" for k, v in loss_dict_G.items() if v is not None])
        d_str = ', '.join([f"D_{k}: {v:.4f}" for k, v in loss_dict_D.items() if v is not None])
        print(f"[{step}/{epoch}] {g_str} | {d_str}")
        
        global_step += 1
    
    # validate and save
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        model.set_input(val_source_img, val_target_img, val_gtwarped_img)
        val_gen_y, val_warp_result = model.predict()
        val_warped_x, val_pos_field = val_warp_result
        val_warped_x_np = ((val_warped_x[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        val_gen_y_np = ((val_gen_y[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        val_source_img_np = ((val_source_img[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        val_target_img_np = ((val_target_img[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        val_gtwarped_img_np = ((val_gtwarped_img[0].permute((1, 2, 0)).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)

        save_path = os.path.join(args.model_dir, f"{args.exp_name}-epoch{epoch}")
        os.makedirs(save_path, exist_ok=True)
        val_save_path = os.path.join(save_path, "validation")
        os.makedirs(val_save_path, exist_ok=True)
        cv2.imwrite(os.path.join(val_save_path, f"val_{val_img_index}_warped_x.png"), val_warped_x_np)
        cv2.imwrite(os.path.join(val_save_path, f"val_{val_img_index}_gen_y.png"), val_gen_y_np)
        cv2.imwrite(os.path.join(val_save_path, f"val_{val_img_index}_gt_warped.png"), val_gtwarped_img_np)
        cv2.imwrite(os.path.join(val_save_path, f"val_{val_img_index}_source_img.png"), val_source_img_np)
        cv2.imwrite(os.path.join(val_save_path, f"val_{val_img_index}_target_img.png"), val_target_img_np)
        
        plot_deformation_grid(val_pos_field, os.path.join(val_save_path, f"val_{val_img_index}_deform_grid.png"))
        plot_deformation_heatmap(val_pos_field, os.path.join(val_save_path, f"val_{val_img_index}_deform_heatmap.png"))
        
        model.save(save_path)