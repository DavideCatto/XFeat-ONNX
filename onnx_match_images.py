import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

from utils import draw_matches


def main():
    # Setting variables
    dense = True  # Dense keypoints extraction
    multiscale = False  # Dense mode: enable multiscale

    # Get image and load
    fname_img_ref = "assets/ref.png"
    fname_img_curr = "assets/tgt.png"
    img_ref = cv2.imread(fname_img_ref)
    img_curr = cv2.imread(fname_img_curr)

    # Create models
    fname_model = "weights/xfeat.onnx"
    fname_match_model = "weights/matching.onnx"

    # Check dense
    if dense:
        fname_model = fname_model.replace(".onnx", "_dense.onnx")
        fname_match_model = fname_match_model.replace(".onnx", "_dense.onnx")

    # Create Extractor Model
    session_ext = onnxruntime.InferenceSession(fname_model)
    input_ext_names = session_ext.get_inputs()[0].name
    output_ext_names = [node.name for node in session_ext.get_outputs()]

    # Create Matching Model
    session_match = onnxruntime.InferenceSession(fname_match_model)
    input_match_names = [node.name for node in session_match.get_inputs()]
    output_match_names = [node.name for node in session_match.get_outputs()]

    # Convert to tensor
    img_ref_RGB = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_curr_RGB = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)

    # Parse numpy array to tensor
    img_ref_tensor = np.array([img_ref_RGB.transpose(2, 0, 1)], dtype=np.float32) / 255.0
    img_curr_tensor = np.array([img_curr_RGB.transpose(2, 0, 1)], dtype=np.float32) / 255.0

    # Run Extractor model
    out_ref = session_ext.run(output_ext_names, {input_ext_names: img_ref_tensor})
    out_curr = session_ext.run(output_ext_names, {input_ext_names: img_curr_tensor})

    # Input tensor
    input_tensor = {
        input_match_names[0]: out_ref[0],
        input_match_names[1]: out_ref[1],
        input_match_names[2]: out_curr[0],
        input_match_names[3]: out_curr[1],
    }

    if dense:
        input_tensor.update({input_match_names[4]: out_ref[2]})

    # Run Matching model
    mkpts0, mkpts1 = session_match.run(output_match_names, input_tensor)

    # Draw matches
    img_matches = draw_matches(mkpts0, mkpts1, img_ref, img_curr)

    # Show
    plt.figure(figsize=(20, 20))
    plt.imshow(img_matches[..., ::-1])
    plt.show()

    # Save if you want
    cv2.imwrite(os.path.join(os.path.dirname(fname_img_ref), "match" + ("_dense" if dense else "") + "_onnx.png"), img_matches)

if __name__ == "__main__":
    main()
