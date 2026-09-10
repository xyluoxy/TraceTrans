import os
import sys
import glob
import numpy as np
import cv2
import json

# dataloader
def img_to_img(
    flist_path,
    batch_size=1,
    output_shape=[256,256],
    use_rgb = False,
    use_depth=False
):
    """
    Infinite generator of image-to-image training batches.

    Parameters:
        flist_path: path to a JSON file list. Its ``"dataset"`` entries provide, per
            sample, an ``idx`` plus ``before`` / ``after`` / ``gt_warped`` image paths
            (and ``before_depth`` / ``after_depth`` when ``use_depth`` is set), all
            relative to the directory containing ``flist_path``.
        batch_size: number of samples per yielded batch. Default 1.
        output_shape: ``(width, height)`` each image is resized to. Default ``[256, 256]``.
        use_rgb: load images as 3-channel RGB instead of single-channel grayscale.
        use_depth: also load the matching depth maps and concatenate them as an extra
            channel on the source and target images.

    Yields:
        tuple ``(img_indices, source_img_batch, target_img_batch, gt_warped_img_batch)``
        where ``img_indices`` is the list of sample ``idx`` values and the three batches
        are ``float32`` arrays of shape ``[B, H, W, C]`` with intensities scaled to
        ``[-1, 1]``.
    """
    if(not os.path.isfile(flist_path)):
        raise Exception(f"flist '{flist_path}' is not a valid file")
    elif(flist_path[-4:] != 'json'):
        raise Exception(f"flist '{flist_path}' is not a json file") 

    with open(flist_path, "r") as f:
        flist = json.load(f)

    dataset_root_path = os.path.dirname(flist_path)
    
    def load_image(file_path, use_rgb=False):
        """Helper function to load and process a single image"""
        img = cv2.imread(os.path.join(dataset_root_path, file_path))
        if use_rgb:
            img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, dsize=tuple(output_shape), interpolation=cv2.INTER_LINEAR)
        
        if not use_rgb:
            img = img[..., None]
        
        return img.astype(np.float32) / 127.5 - 1.0
    
    while(True):
        files = flist["dataset"]
        
        target_indices = np.random.randint(len(files), size=batch_size)
        
        source_indices = target_indices
        target_files = [files[i] for i in target_indices]
        source_files = [files[i] for i in source_indices]
        
        img_indices = [f["idx"] for f in source_files]
        
        target_imgs = [load_image(f["after"], use_rgb) for f in target_files]
        source_imgs = [load_image(f["before"], use_rgb) for f in source_files]
        gt_warped_imgs = [load_image(f["gt_warped"], use_rgb) for f in source_files]
        
        if use_depth:
            target_depths = [load_image(f["after_depth"], use_rgb=False) for f in target_files]
            target_imgs = np.concatenate([target_imgs, target_depths], axis=-1)
            source_depths = [load_image(f["before_depth"], use_rgb=False) for f in source_files]
            source_imgs = np.concatenate([source_imgs, source_depths], axis=-1)
        
        target_img_batch = np.stack(target_imgs, 0)
        source_img_batch = np.stack(source_imgs, 0)
        gt_warped_img_batch = np.stack(gt_warped_imgs, 0)

        yield (img_indices, source_img_batch, target_img_batch, gt_warped_img_batch)

