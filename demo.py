import cv2
import numpy as np

from mmpose.apis import MMPoseInferencer

# Load a pretrained human pose model (we'll adapt later)
inferencer = MMPoseInferencer('human')

# Load your image
img_path = 'test.jpg'   # <-- replace with your image
img = cv2.imread(img_path)

# Run inference
results = inferencer(img_path, show=False)

# Extract predictions
result = next(results)
predictions = result['predictions'][0]

# Draw keypoints
for person in predictions:
    keypoints = person['keypoints']

    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

# Show result
cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()