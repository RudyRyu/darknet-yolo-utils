# Python utils for Darknet YOLO

- Infer (OpenCV DNN module)
- Detect (detect image, detect video, detect video with ROI)
  - single inference
  - single inference with multiprocessing
  - batch inference
- Evaluate
  - show which images raise detection error (cannot know with Darknet)
- Get Anchors Boxeswith Kmeans clustering 
  - input with Darknet data format
  - reference: https://github.com/david8862/keras-YOLOv3-model-set
- Generate Training Images
  - image augmentation pipeline only for digit detection
- Convert Darknet weights to Tensorflow
  - only for custom models (not for official YOLO models)
  - reference: https://github.com/hunglc007/tensorflow-yolov4-tflite
  - reference: https://github.com/zzh8829/yolov3-tf2