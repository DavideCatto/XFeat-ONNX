import os
import cv2
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt

from module.xfeat import XFeat
from utils import draw_points, load_image, draw_matches


def main():
    # Setting variables
    dense = True                    # Dense keypoints extraction
    multiscale = False              # Dense mode: enable multiscale

    # Get image and load
    fname_img_ref = "assets/ref.png"
    fname_img_curr = "assets/tgt.png"
    img_ref = cv2.imread(fname_img_ref) # For debug purpose
    img_curr = cv2.imread(fname_img_curr) # For debug purpose

    # Create model
    fname_model = "weights/xfeat.pt"
    xfeat = XFeat(weights=fname_model, top_k=4096, multiscale=dense and multiscale)

    # Parse numpy array to tensor
    img_tensor_ref, _ = load_image(fname_img_ref)
    img_tensor_curr, _ = load_image(fname_img_curr)

    # Run model
    if dense:
        mkpts0, mkpts1 = xfeat.match_xfeat_star(img_tensor_ref, img_tensor_curr)
    else:
        mkpts0, mkpts1 = xfeat.match_xfeat(img_tensor_ref, img_tensor_curr)
    img_matches = draw_matches(mkpts0, mkpts1, img_ref, img_curr)

    # Show
    plt.figure(figsize=(20, 20))
    plt.imshow(img_matches[..., ::-1])
    plt.show()

    # Save if you want
    cv2.imwrite(os.path.join(os.path.dirname(fname_img_ref), "match" + ("_dense" if dense else "") + ".png"), img_matches)


if __name__ == "__main__":
    main()