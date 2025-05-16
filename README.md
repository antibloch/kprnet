# [KPRNet: Improving projection-based LiDARsemantic segmentation](https://arxiv.org/pdf/2007.12668.pdf)

![Video](kprnet.gif)

## Installation

Install [apex](https://github.com/NVIDIA/apex) and the packages in requirements.txt

## Experiment 

Download pre-trained [resnext_cityscapes_2p.pth](https://drive.google.com/file/d/1aioKjoxcrfqUtkWQgbo64w8YoLcVAW2Z/view?usp=sharing). The path should be given in `model_dir`.  CityScapes pretraining will be added later.

The result from paper is trained on 8 16GB GPUs (total batch size 24).

To train run:

```bash
python train_kitti.py \
  --semantic-kitti-dir path_to_semantic_kitti \
  --model-dir location_where_your_pretrained_model_is \
  --checkpoint-dir your_output_dir


# e.g: python train_kitti.py --semantic-kitti-dir kitti_dataset --model-dir pretrained_model  --checkpoint-dir output_kitti
# where kitti_dataset contains dataset/sequences/00..
```

```bash
python run_inference.py --semantic-kitti-dir kitti_dataset/dataset/sequences --output-path pred_stuff  --checkpoint-path pretrained_total_model/kpr_trained.pth
```

```bash
# Velodyne Scan
python run_inference_new.py --checkpoint_path pretrained_total_model/kpr_trained.pth --point_folder data --output_path points_segment --W 2048 --H 64 --fov_up 2.0 --fov_down -24.9 --ring_major --visualize --inverted_depth

# Ouster Scan
python run_inference_new.py --checkpoint_path pretrained_total_model/kpr_trained.pth --point_folder data --output_path points_segment --W 1024 --H 64 --fov_up 44.07 --fov_down -45.73  --visualize --inverted_depth
```



## SemanticKITT Testing

```code
python run_inference.py --semantic-kitti-dir kitti_ds/dataset/sequences --output-path pred_stuff  --checkpoint-path checkpoints/epoch10.pth
python view_inference.py --points kitti_ds/dataset/sequences/08/velodyne --labels kitti_ds/dataset/sequences/08/labels --predictions pred_stuff/sequences/08/predictions --results results
```



## Ouster Testing

```code
python run_inference_new_labelled.py --point_folder refined_data/pc --labels_folder refined_data/label --output_path pred_stuff_ouster  --checkpoint_path checkpoints/epoch10.pth
python view_inference_new_labelled_allscans.py --points pred_stuff_ouster_points --labels pred_stuff_ouster_labels --predictions pred_stuff_ouster_predictions --results results
```

The fully trained model weights can be downloaded [here](https://drive.google.com/file/d/11mUMdFPNT-05lC54Ru_2OwdwqTPV4jrW/view?usp=sharing) .


![output_video-ezgif com-cut](https://github.com/user-attachments/assets/d80eb03a-e5b7-4a3f-b142-c3aaabe7c699)



## Acknowledgments
[KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) 

[RangeNet++](https://github.com/PRBonn/lidar-bonnetal) 

[HRNet](https://github.com/HRNet)

## Reference

KPRNet appears in ECCV workshop Perception for Autonomous Driving.

```
@article{kochanov2020kprnet,
  title={KPRNet: Improving projection-based LiDAR semantic segmentation},
  author={Kochanov, Deyvid and Nejadasl, Fatemeh Karimi and Booij, Olaf},
  journal={arXiv preprint arXiv:2007.12668},
  year={2020}
}
```
