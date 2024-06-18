import numpy as np
import cv2
import onnx
import onnxruntime
from onnxconverter_common import float16
import time
from tqdm.rich import tqdm


def main():
    # TODO! Developing
    fname_img_ref = "assets/ref.png"
    fname_img_curr = "assets/tgt.png"
    fname_model = "weights/xfeat_e2e.onnx"
    num_test = 10

    # Load model
    model = onnx.load(fname_model)

    # Convert to tensor
    img_ref = cv2.imread(fname_img_ref)
    img_curr = cv2.imread(fname_img_curr)
    img_ref_RGB = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_curr_RGB = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)

    # Parse numpy array to tensor
    img_ref_tensor = np.array([img_ref_RGB.transpose(2, 0, 1)], dtype=np.float16) / 255.0
    img_curr_tensor = np.array([img_curr_RGB.transpose(2, 0, 1)], dtype=np.float16) / 255.0

    # Convert model
    fname_model_fp16 = fname_model.replace(".onnx", "_fp16.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, fname_model_fp16)

    # Create Fp16 Model
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_profiling = True
    session = onnxruntime.InferenceSession(fname_model_fp16, sess_options, poviders=["CUDAExecutionProvider"])

    # Get input output nodes
    input_names = [node.name for node in session.get_inputs()]
    output_names = [node.name for node in session.get_outputs()]

    output = session.run(output_names, {input_names[0]: img_ref_tensor, input_names[1]: img_curr_tensor})  # warmup
    start_time = time.time()
    for _ in tqdm(range(num_test)):
        output = session.run(output_names, {input_names[0]: img_ref_tensor, input_names[1]: img_curr_tensor})
    duration = time.time() - start_time
    print(f"Fp16: {(duration * 1000.0)/(float)(num_test)}[ms]")
    prof_file_fp16 = session.end_profiling()


if __name__ == "__main__":
    main()