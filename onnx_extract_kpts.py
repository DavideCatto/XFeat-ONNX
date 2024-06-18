import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

from utils import draw_points, load_image


def main():
    # Setting variables
    dense = True  # Dense keypoints extraction
    multiscale = False  # Dense mode: enable multiscale

    # Get image and load
    fname_img = "assets/ref.png"
    img = cv2.imread(fname_img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create model
    fname_model = "weights/xfeat.onnx"
    if dense:
        fname_model = fname_model.replace(".onnx", "_dense.onnx")

    session = onnxruntime.InferenceSession(fname_model)
    input_names = session.get_inputs()[0].name
    output_names = [node.name for node in session.get_outputs()]

    # Parse numpy array to tensor
    img_tensor = np.array([imgRGB.transpose(2, 0, 1)], dtype=np.float32) / 255.0

    # Run model
    results = session.run(output_names, {input_names: img_tensor})
    img = draw_points(img, results[0])

    # Show
    plt.figure(figsize=(20, 20))
    plt.imshow(img[..., ::-1])
    plt.show()

    # Save if you want
    cv2.imwrite(fname_img.replace(".", "_res_onnx."), img)


if __name__ == "__main__":
    main()