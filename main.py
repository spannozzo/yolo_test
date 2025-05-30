from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# # Train the model on the COCO8 dataset for 100 epochs
# train_results = model.train(
#     data="coco8.yaml",  # Path to dataset configuration file
#     epochs=100,  # Number of training epochs
#     imgsz=640,  # Image size for training
#     device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
# )

# Evaluate the model's performance on the validation set
# metrics = model.val()

# Perform object detection on an image


# Export the model to ONNX format for deployment
# path = model.export(format="onnx") 
# print(path)# Returns the path to the exported model

import time

# Record start time
start_time = time.time()

# Perform prediction
results = model("image2.png")  # Predict on an image

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Prediction took {elapsed_time:.2f} seconds")

results[0].show()  # Display results