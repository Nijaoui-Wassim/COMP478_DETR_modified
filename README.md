# Modified DETR for Object Detection on MiniCOCO - End-to-End Object Detection with Transformers

This project presents a modified architecture of the Detection Transformer (DETR) trained from scratch on the MiniCOCO dataset. The DETR model, originally designed for the full COCO dataset, has been adapted to leverage the smaller and more efficient MiniCOCO for quicker experimentation and resource-efficient training.

## Modifications to DETR Architecture

The architecture of DETR has been modified to improve performance and efficiency on the MiniCOCO dataset. Key changes include:

- **Enhanced Backbone:** Modifications to the backbone network to better accommodate the characteristics of the MiniCOCO dataset.
- **Transformer Adjustments:** Changes in the transformer configuration to optimize for the reduced size and variance of MiniCOCO.
- **Output Layers:** Adjustments in the output layer to match the specific requirements of MiniCOCO's annotation style.

Detailed documentation of all changes is available in the `detr_modified.py` file, highlighting the architectural differences aimed at enhancing model performance and efficiency.

## Main Idea
- The core innovation in the modified DETR (Detection Transformer) model lies in its approach to handling the feature extraction phase, a crucial step where raw images are transformed into a lower-dimensional representation that captures essential information. Instead of processing the entire low-dimensional image as a whole, the modified architecture introduces a novel method: serving the image in batches of patches.
- This approach segments the feature-extracted image into smaller, manageable patches, which are then fed into the transformer model in batches. This technique allows the model to focus on localized regions of the image, capturing more detailed and contextual information from each segment. By analyzing these patches both independently and in relation to one another, the model gains a more nuanced understanding of the image content, leading to enhanced object detection capabilities.
- The segmentation of the image into patches enables the transformer to hone in on specific features and relationships within the image, which might be overlooked when viewing the image as a whole. This method is particularly beneficial for complex scenes with multiple objects, intricate backgrounds, or subtle object interactions, where context plays a crucial role in accurate detection.
- the main idea behind the modifications to the DETR model is to leverage the use of sliding windows with image patches as input as an added step, more contextually aware representation of the image. This approach aims to improve the model's ability to detect objects accurately by providing a more comprehensive understanding of the image context, leading to better performance in object detection tasks.

### Original Simplified architecture
![originalDETR](https://github.com/Nijaoui-Wassim/COMP478_DETR_modified/assets/52583856/367a0e6e-d6dc-445d-bb8f-d5bf4e9de732)

### Our Simplified architecture
![ourarch](https://github.com/Nijaoui-Wassim/COMP478_DETR_modified/assets/52583856/c1ce8732-75ba-4ce0-95e8-1d03a6223f41)


## MiniCOCO Dataset Overview

MiniCOCO is a curated subset of the COCO 2017 dataset, designed for hyperparameter tuning and cost-effective ablation studies. It consists of 25,000 images (~20% of the original Train2017 set) with object instance statistics closely matching the full dataset. The MiniCOCO dataset ensures that the proportion of object instances per class, and the ratios of small, medium, and large objects, both overall and per class, are preserved.

For more information and to download the MiniCOCO dataset, refer to the ECCV 2020 paper:
@inproceedings{HoughNet,
author = {Nermin Samet and Samet Hicsonmez and Emre Akbas},
title = {HoughNet: Integrating near and long-range evidence for bottom-up object detection},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020},
}


## Usage

To train the modified DETR model on MiniCOCO, follow these steps:

1. Download the MiniCOCO dataset from the official minicoco dataset by following their steps.
2. Install requirements and setup for DETR dataset based on the original paper repo.
3. Place the dataset in the appropriate directory.
4. Adjust the training configuration in `main_modified.py`, visualization file, the transformer file and so on as needed.
5. Run the training script: `python main_modified.py --dataset_file minicoco`

## Limitations
Although we trained on the minicoco dataset, it still took too long on our hardware and we managed to do few runs between 10 and 30 epochs to get remotely decent results.
Better results should be achievable by training for longer.


## Citation

If you use this modified DETR implementation or the MiniCOCO dataset in your work, please cite the original DETR paper and the ECCV 2020 paper introducing MiniCOCO.

## Acknowledgements

This project builds upon the innovative work by the DETR team and the creators of the MiniCOCO dataset. Special thanks to the authors of the ECCV 2020 paper for making MiniCOCO publicly available.





# Initial Results
#### Some of the results after epoch 19:
![minicoco_epoch_19](https://github.com/Nijaoui-Wassim/COMP478_DETR_modified/assets/52583856/0a93c456-b84c-4dcd-8038-7f1ec4d23958)

#### Some of the results after epoch 30:
![minicoco_30](https://github.com/Nijaoui-Wassim/COMP478_DETR_modified/assets/52583856/d70571df-3404-4e50-a92e-df7898939c7c)
![minicoco_ep30_3](https://github.com/Nijaoui-Wassim/COMP478_DETR_modified/assets/52583856/a7ccd5dd-51ea-4a25-bfa6-da34a6883e52)
![minicoco_ep30_4](https://github.com/Nijaoui-Wassim/COMP478_DETR_modified/assets/52583856/b436799d-de46-4dc9-b564-9f80875ebc8d)

