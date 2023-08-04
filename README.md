# Foreground Object Search by Distilling Composite Image Feature, *ICCV 2023*.

## Requirements

- See requirements.txt for other dependencies.

## Data Preparing

- Download Open-Images-v6 trainset from [Open Images V6 - Download](https://storage.googleapis.com/openimages/web/download_v6.html) and unzip them. We recommend that you use FiftyOne to download the Open-Images-v6 dataset. After the dataset is downloaded, the data structure of Open-Images-v6 dataset should be as follows.
  
  ```
  Open-Images-v6
  ├── metadata
  ├── train
  │   ├── data
  │   │   ├── xxx.jpg
  │   │   ├── xxx.jpg
  │   │   ...
  │   │
  │   └── labels
  │       └── masks
  │       │   ├── 0
  │       │       ├── xxx.png
  │       │       ├── xxx.png
  │       │       ...
  │       │   ├── 1
  │       │   ...
  │       │
  │       ├── segmentations.csv
  │       ...
  ```

- Download S-FOSD annotations, R-FOSD annotations and background images of R-FOSD from [Baidu disk](https://pan.baidu.com/s/1LF_4LbwxbxSBy-zqBkgzDw) (code: 3wvf) and save them to the appropriate location under the `data` directory according to the data structure below. 
  
- Generate backgrounds and foregrounds.
  
  ```
  python prepare_data/fetch_data.py --open_images_dir <path/to/open/images>
  ```

The data structure is like this:

```
data
├── metadata
│   ├── classes.csv
│   └── category_embeddings.pkl
├── test
│   ├── bg_set1
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   ...
│   │
│   ├── bg_set2
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   ...
│   │
│   ├── fg
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   ...
│   └── labels
│       └── masks
│       │   ├── 0
│       │       ├── xxx.png
│       │       ├── xxx.png
│       │       ...
│       │   ├── 1
│       │   ...
│       │
│       ├── test_set1.json
│       ├── test_set2.json
│       └── segmentations.csv
│
└── train
    ├── bg
    │   ├── xxx.jpg
    │   ├── xxx.jpg
    │   ...
    │
    ├── fg
    │   ├── xxx.jpg
    │   ├── xxx.jpg
    │   ...
    │
    └── labels
        └── masks
        │   ├── 0
        │       ├── xxx.png
        │       ├── xxx.png
        │       ...
        │   ├── 1
        │   ...
        │
        ├── train_sfosd.json
        ├── train_rfosd.json
        ├── category.json
        ├── number_per_category.csv
        └── segmentations.csv
```

## Pretrained Model

We provide the checkpoint ([Baidu disk](https://pan.baidu.com/s/1_Dh2w08AAqdsw8Cb3l4nfQ) code: 7793) for the evaluation on S-FOSD dataset and checkpoint ([Baidu disk](https://pan.baidu.com/s/17jq1FWKSsEngp7scB4357Q) code: 6kme) for testing on R-FOSD dataset. By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

## Testing

### Evaluation on S-FOSD Dataset

```
python evaluate/evaluate.py --testOnSet1
```

### Evaluation on R-FOSD Dataset

```
python evaluate/evaluate.py --testOnSet2
```

The evaluation results will be stored to the directory `eval_results`.

If you want to save top 20 results on R-FOSD, add `--saveTop20 ` parameter. The top 20 results on R-FOSD will be stored to the directory `top20` by default.

If you want to save the model's prediction scores on R-FOSD, add `--saveScores` parameter. The model scores on R-FOSD will be stored to the directory `model_scores` by default.

## Training

Please download the pretrained teacher models from [Baidu disk](https://pan.baidu.com/s/1D_zT326PLXZ-C0j5mcCY6A) (code: 40a5) and save the model to directory `checkpoints/teacher`. 

To train a new sfosd model, you can simply run:

```
.train/train_sfosd.sh
```

Similarly, train a new rfosd model by:

```
.train/train_rfosd.sh
```

## License

Both background and foreground images of S-FOSD belong to Open-Images. The background images of R-FOSD are collected from Internet and are licensed under a Creative Commons Attribution 4.0 License.