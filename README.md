# NanoChimp: lightweight detection, tracking and reid of chimpanzees

## Requirements

- Python 3.8+
- CUDA GPU recommended

## Clone repos and create conda env

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/NanoChimp.git
cd NanoChimp

conda create -n nanochimp python=3.10 -y
conda activate nanochimp
pip install -r requirements.txt # might need CUDA version of torch 
```

## Your repos should like this

```
configs/
  train.py          # Train YOLO detection model
  train_reid.py     # Train Re-ID model
  inference.py      # Run detection, tracking, and Re-ID on video
lib/
  reid_model.py     # Re-ID network definition
tools/              # Data prep, gallery building, evaluation
bytetrack.yaml      # Tracker config (default)
botsort.yaml        # Alternative tracker config
```

## Pipeline

1. **Detection data annotation and preparation** — annotate in CVAT, convert to COCO, merge batches
2. **Train detection model** — YOLO bounding-box model
3. **Prepare Re-ID data** — crop chimps into identity folders under `data/reid/`
4. **Train Re-ID model** — ResNet-based embedding model
5. **Build gallery** — reference embeddings for known individuals
6. **Run inference** — detection + tracker + named ID assignment

## Detection data

Training data are not included in this repository.

I annotated my detection data in CVAT and exported the data from CVAT using 'CVAT for Images 1.1'. I then converted the annotations to COCO using `tools/cvat_to_coco.py`. The COCO annotations are then converted to YOLO format in the `configs/train.py` script automatically.  

Place your own detection data as follows:

- `data/images/` — detection training images  
- `data/annotations/` — COCO-format JSON annotations

## Train detection

Specify the YOLO model to train, the base weights are added to wd automtically. Currently set to YOLOv11n.

```bash
python configs/train.py
```

## Re-ID data

For the ReID model you need to create a library of cropped images for each ID in your population and store them here: 

- `data/reid/<identity_name>/` — Re-ID crop images per animal

The below script can be used to create the cropped ID images using your trained detection model. But you have to sort them into the `data/reid/` folders.

```bash
python tools/extract_crops_for_labelling.py
```

## Train Re-ID

Train ReID model using a ResNet-50 backbone using a custom ReIDNet from `lib/reid_model.py`. 

```bash
python tools/split_reid_dataset.py
python configs/train_reid.py
python tools/build_reid_gallery.py
```

## Configure tracking config

Use either the botsort.yaml or the bytertrack.yaml in the wd to configure the tracking algorithm (specify the tracker in the `configs/inference.py`)

## Run inference (detection + tracking + Re-ID)

Pop the images or videos in the `infer_input` folder

```bash
python configs/inference.py --use_tracking --use_reid --input_dir infer_input --output_dir infer_output
```

## Evaluation

```bash
python tools/evaluate_models.py        # detection metrics
python tools/evaluate_reid_model.py    # Re-ID metrics
python tools/evaluate_tracking.py      # tracking metrics (MOT ground truth required*)
```

*Multiple Object Tracking (MOT) ground truth is a fully annotated test video. For every frame in the video, each chimp has a accurate bounding box and a persistent track ID that is assigned to that individual through the video. This is what lets you measure tracking performance. To create it, annotate a video in CVAT (track each chimp across frames), export as "MOTS 1.1", and use the `gt/gt.txt` file inside the zip. Run inference with `--use_tracking` to produce a matching tracker output `.txt` then compare the two with the evaluation script, e.g.

```bash
python tools/evaluate_tracking.py --gt path/to/gt.txt --ts infer_output/tracking_<video>_<timestamp>.txt
```

## Citation

If you use this code, please cite:

```
Lewton, J. (2026). NanoChimp: Lightweight detection, tracking, and re-identification of chimpanzees (Version 1.0.0). GitHub. https://github.com/JackLewton/NanoChimp
```

## License

This poject is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.