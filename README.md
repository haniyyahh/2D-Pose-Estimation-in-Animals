CS 6384 Project: 2D Pose Estimation in Animals

In this project, we investigated applying human 2D pose estimation frameworks to animal 2D pose estimation frameworks.
First, the files MediaPipe_zeroshot and yolopose contain code for the initial zeroshot experiments.
Next, the files MediaPipe_version2 and YOLOv8_version2 contain code for the fine-tuned cross-domain transfer learning code
(changing the keypoint mapping).
Finally, the file YOLOv8_Custom_Geometric_Loss applies a custom geometric prior to penalize the model for producing skeletal inconsistencies.
