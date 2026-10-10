# Methodology



This project investigates pixel-level segmentation

of defects in industrial tube inner-wall images.



The baseline uses a ResNet34-encoder U-Net.



Inference is performed at 512 x 512 resolution,

and probability maps are resized to the original

image dimensions for pixel-level evaluation.



Fine-tuning explores hard-example sampling,

including missed defects and background regions.



See the research report for detailed experimental

protocols and results.
