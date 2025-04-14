# Object-Aware Crop
## Intorduction
Many object detection applications require the creation of high-resolution datasets with detailed bounding box 
annotations. However, the large spatial dimensions of these images pose challenges for efficient deep learning 
training, both in terms of memory consumption and computational cost.

A common solution is to downsample high-resolution images to smaller sizes. While this reduces memory requirements, 
it often leads to the loss of critical details—especially problematic for tasks involving small or fine-grained 
objects. An alternative strategy is to crop large images into smaller crops. However, most existing cropping 
algorithms are agnostic to object locations and may inadvertently divide objects across crop boundaries. This can 
degrade detection performance, particularly in applications where preserving full object context is crucial (e.g., medical 
imaging, autonomous driving, and precision agriculture).

To address this issue, we introduce Object-Aware Crop (OA Crop), a novel cropping algorithm that generates small image 
crops without cutting through objects or their bounding boxes. OA Crop ensures that each crop contains either complete 
objects or none at all, making it highly suitable for scenarios where object integrity must be preserved. Our method 
enables effective training on high-resolution datasets while maintaining the semantic completeness of annotated objects.

## Algorithm
The overall algorithm of OA Crop is illustrated as follow:
![profile](/imgs/oac.png)
The figure below shows the idea of generating the range x_start, x_end, y_start, y_end of the top-left coordinate (xp, yp) 
of a crop that can fully cover a given bounding box.
![profile](/imgs/crop_range.png)

## Usage
Download the "object_aware_crop.py", put it into a suitable folder in your project. Refer to "demo.py" for the usage. OA Crop
only support YOLO format labels. In your label file, for each bounding box you should have something like this "class_label, x_center, y_center, width, height", 
"class label" indicate the class of the object within the bounding box. x_center and y_center are the normalized coordinates of the center of the bounding box. 
The width and height are the normalized length. 

