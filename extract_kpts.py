import cv2
import matplotlib.pyplot as plt

from module.xfeat import XFeat
from utils import draw_points, load_image


def main():
    # Setting variables
    dense = True                    # Dense keypoints extraction
    multiscale = False              # Dense mode: enable multiscale

    # Get image and load
    fname_img = "assets/ref.png"
    img = cv2.imread(fname_img)     # For debug purpose

    # Create model
    fname_model = "weights/xfeat.pt"
    xfeat = XFeat(weights=fname_model, top_k=4096)

    # Parse numpy array to tensor
    img_tensor, _ = load_image(fname_img)

    # Run model
    if dense:
        results = xfeat.detectAndComputeDense(img_tensor, multiscale=multiscale)
    else:
        results = xfeat.detectAndCompute(img_tensor)
    img = draw_points(img, results["keypoints"])

    # Show
    plt.figure(figsize=(20, 20))
    plt.imshow(img[..., ::-1])
    plt.show()

    # Save if you want
    cv2.imwrite(fname_img.replace(".", "_res."), img)


if __name__ == "__main__":
    main()