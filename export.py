import argparse
import os.path
from typing import List
import torch

from module_onnx.xfeat import XFeat
# from lightglue_onnx.end2end import normalize_keypoints
from utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xfeat_path",
        type=str,
        default="weights/xfeat.pt",
        required=False,
        help="Path to load the feature extractor PT model.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the feature extractor ONNX model.",
    )

    parser.add_argument(
        "--dynamic", action="store_true", help="Whether to allow dynamic image sizes."
    )

    parser.add_argument(
        "--dense",
        action="store_true",
        help="Whether to export a dense keypoints extractor instead of simple extractor.",
    )

    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )

    # Extractor-specific args:
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        required=False,
        help="Maximum number of keypoints outputted by the extractor.",
    )

    return parser.parse_args()


def export_onnx(
    xfeat_path=None,
    output_path=None,
    img0_path="assets/ref.png",
    img1_path="assets/tgt.png",
    dynamic=False,
    dense=False,
    end2end=False,
    top_k=None,
):
    # Sample images for tracing
    image0, _ = load_image(img0_path)
    image1, _ = load_image(img1_path)

    # Models
    xfeat = XFeat(weights=xfeat_path, top_k=top_k).eval()

    # ONNX Export
    if end2end:
        # ------------------------------
        # Export Extractor and Matching
        # ------------------------------
        output_path = xfeat_path.replace(".pt", "_e2e.onnx")
        xfeat.forward = xfeat.match_xfeat

        if dense:
            xfeat.forward = xfeat.match_xfeat_star
            output_path = xfeat_path.replace(".pt", "_dense_e2e.onnx")

        dynamic_axes = {
            "mkpts0": {0: "num_keypoints"},
            "mkpts1": {0: "num_keypoints"},
        }
        if dynamic:
            dynamic_axes.update({"image0": {2: "height", 3: "width"},
                                 "image1": {2: "height", 3: "width"}})
        else:
            print(
                f"WARNING: Exporting without --dynamic implies that the extractor's input image size will be locked to {image0.shape[-2:]}"
            )
            output_path = output_path.replace(
                ".onnx", f"_{image0.shape[-2]}x{image0.shape[-1]}.onnx"
            )

        torch.onnx.export(
            xfeat,
            (image0, image1),
            output_path,
            verbose=False,
            do_constant_folding=True,
            input_names=["image0", "image1"],
            output_names=["mkpts0", "mkpts1"],
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )

    else:
        # -----------------
        # Export Extractor
        # -----------------
        dynamic_axes = {
            "keypoints": {0: "num_keypoints"},
            "descriptors": {0: "num_keypoints"}
        }

        if dense:
            output_path = xfeat_path.replace(".pt", "_dense.onnx")
            xfeat.forward = xfeat.detectAndComputeDense
            dynamic_axes.update({"scales": {0: "num_keypoints"}})
            output_names = ["keypoints", "descriptors", "scales"]
        else:
            output_path = xfeat_path.replace(".pt", ".onnx")
            xfeat.forward = xfeat.detectAndCompute
            dynamic_axes.update({"scores": {0: "num_keypoints"}})
            output_names = ["keypoints", "descriptors", "scores"]

        # Add dynamic input
        if dynamic:
            dynamic_axes.update({"images": {2: "height", 3: "width"}})
        else:
            print(
                f"WARNING: Exporting without --dynamic implies that the extractor's input image size will be locked to {image0.shape[-2:]}"
            )
            output_path = output_path.replace(
                ".onnx", f"_{image0.shape[-2]}x{image0.shape[-1]}.onnx"
            )

        # Export model
        torch.onnx.export(
            xfeat,
            image0,
            output_path,
            verbose=False,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )

        # -----------------
        # Export Matching
        # -----------------

        # Simulate keypoints, features
        kpts = torch.rand(top_k, 2, dtype=torch.float32)
        descr = torch.rand(top_k, 64, dtype=torch.float32)
        scales = torch.rand(top_k, dtype=torch.float32)

        # Dynamic input
        dynamic_axes = {
            "kpts0": {0: "num_kpts0"},
            "feats0": {0: "num_kpts0"},
            "kpts1": {0: "num_kpts1"},
            "feats1": {0: "num_kpts1"},
        }

        input_names = ["kpts0", "feats0", "kpts1", "feats1"]
        input_values = [kpts, descr, kpts, descr]
        if dense:
            output_matching_path = os.path.join(os.path.dirname(output_path), "matching_dense.onnx")
            xfeat.forward = xfeat.match_star_onnx
            dynamic_axes.update({"scales0": {0: "num_kpts0"},})
            input_names.append("scales0")
            input_values.append(scales)
        else:
            output_matching_path = os.path.join(os.path.dirname(output_path), "matching.onnx")
            xfeat.forward = xfeat.match_onnx

        torch.onnx.export(
            xfeat,
            tuple(input_values),
            output_matching_path,
            verbose=False,
            do_constant_folding=False,
            input_names=input_names,
            output_names=["mkpts0", "mkpts1"],
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
