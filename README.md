# Foreground Object Search Dataset FOSD

This is the official repository for the following paper:

> **Foreground Object Search by Distilling Composite Image Feature**  [[arXiv]](https://arxiv.org/pdf/2308.04990.pdf)<br>
>
> Bo Zhang, Jiacheng Sui, Li Niu<br>
> Accepted by **ICCV 2023**.

**Our model has been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try ＼(^▽^)／** 

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

- Download S-FOSD annotations, R-FOSD annotations and background images of R-FOSD from [Baidu disk](https://pan.baidu.com/s/1LF_4LbwxbxSBy-zqBkgzDw) (code: 3wvf) or [Dropbox](https://www.dropbox.com/scl/fo/hspmvxbz80trifnj299k1/AMPBdBHkU_J1gKova3abqwc?rlkey=fms6xdspj8a8lshh2iw905mih&st=meaj1v5q&dl=0) and save them to the appropriate location under the `data` directory according to the data structure below. 
  
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

We provide the checkpoint ([Baidu disk](https://pan.baidu.com/s/1_Dh2w08AAqdsw8Cb3l4nfQ) code: 7793) or [Dropbox](https://www.dropbox.com/scl/fi/m8ek5m62jw2f22egi18sw/sfosd.pth?rlkey=qnkqxlu3tr1jqw2xefpkljsdu&st=ru95xdkr&dl=0) for the evaluation on S-FOSD dataset and checkpoint ([Baidu disk](https://pan.baidu.com/s/17jq1FWKSsEngp7scB4357Q) code: 6kme)  or [Dropbox](https://www.dropbox.com/scl/fi/7f278c270wbeop1i387wj/rfosd.pth?rlkey=kocu38egtqr9d0bings5toye8&st=4a9edban&dl=0) for testing on R-FOSD dataset. By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

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

Please download the pretrained teacher models from [Baidu disk](https://pan.baidu.com/s/1D_zT326PLXZ-C0j5mcCY6A) (code: 40a5) or [Dropbox](https://www.dropbox.com/scl/fo/engplb7s60urs89g6vegc/APzYeC9H-T4mHCQdk3YzNJA?rlkey=qt1em79qmwmv2e5fr6lkfuz7i&st=ynfabtm4&dl=0) and save the model to directory `checkpoints/teacher`. 

To train a new sfosd model, you can simply run:

```
.train/train_sfosd.sh
```

Similarly, train a new rfosd model by:

```
.train/train_rfosd.sh
```

## FOS Score

Our model can be used to evaluate the compatibility between foreground and background in terms of geometry and semantics.

To launch the demo, you can run:

```
python demo/demo_ui.py
```

Here are three steps you can take to get a compatibility  score for the foreground and the background.

1) Upload a background image in the left box of the first row

2) Click the left-top point and the right-bottom point of the bounding box in the right box of the first row

3) Upload a foreground image in the left box of the second row, then click 'run' button.

## Other Resources

+ [Awesome-Foreground-Object-Search](https://github.com/bcmi/Awesome-Foreground-Object-Search)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)

## License

Both background and foreground images of S-FOSD belong to Open-Images. The background images of R-FOSD are collected from Internet and are licensed under a Creative Commons Attribution 4.0 License.
