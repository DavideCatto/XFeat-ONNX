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
    fname_model = "weights/xfeat_e2e.onnx"
    if dense:
        fname_model = fname_model.replace("e2e", "dense_e2e")

    # Create Extractor and Matching Model
    session = onnxruntime.InferenceSession(fname_model)
    input_names = [node.name for node in session.get_inputs()]
    output_names = [node.name for node in session.get_outputs()]

    # Convert to tensor
    img_ref_RGB = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_curr_RGB = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)

    # Parse numpy array to tensor
    img_ref_tensor = np.array([img_ref_RGB.transpose(2, 0, 1)], dtype=np.float32) / 255.0
    img_curr_tensor = np.array([img_curr_RGB.transpose(2, 0, 1)], dtype=np.float32) / 255.0

    # Run Extractor model
    mkpts0, mkpts1 = session.run(output_names, {
        input_names[0]: img_ref_tensor, input_names[1]: img_curr_tensor})

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
