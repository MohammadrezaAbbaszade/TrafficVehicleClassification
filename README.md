# Traffic Vehicle Classification

An eight-class classifier for cropped traffic-camera vehicle images. The project combines a careful data audit, controlled CNN experiments, ResNet18 transfer learning, error analysis, and a prediction interface that can flag uncertain cases for human review.

**Selected model:** ImageNet-pretrained ResNet18 with `layer4` and the classification head fine-tuned. On a fixed, group-safe **internal validation** set it correctly classified **833/867 images (96.1% accuracy, 0.961 macro-F1)**. A separate mentor-held test set is unavailable here; no hidden-test score is claimed.

## What the system does

- Predicts one of `ambulance`, `autobus`, `kamyun`, `kamyunet`, `minibus`, `savari`, `taxi`, or `vanet` from a **vehicle crop**.
- Returns the predicted class, softmax scores for all eight classes, the highest score, and a `needs_review` flag.
- Supports both a JSON command-line predictor and a FastAPI file-upload endpoint.
- Treats the supplied `neysan` examples as part of the broader `vanet` target class, keeping the output taxonomy at eight classes.

This is a crop classifier, not a vehicle detector for full traffic scenes. The confidence value is a model score, not a calibrated guarantee that the prediction is correct.

## Data and evaluation protocol

The source images are private and are **not included in this repository**. Two supplied data batches were used for development. Following the updated evaluation guidance, the earlier supplied `test` folder could be included in the development pool; the actual final test set is held separately by the evaluators and was never used for model selection.

The audit inventoried **4,360 images** and removed **24 exact extra copies**, leaving **4,336 eligible unique images**. Content hashes identify exact duplicates even when filenames differ. Perceptual hashes group near-duplicates before splitting. A fixed seed (`42`) and `StratifiedGroupKFold` create a development split that separates near-duplicate groups while retaining representation across class and source:

| Source | Training | Validation |
|---|---:|---:|
| Source-labelled images | 1,865 | 467 |
| `unclean` images | 1,405 | 350 |
| `neysan` images, mapped to `vanet` | 199 | 50 |
| **Total** | **3,469** | **867** |

Labels from `unclean` remain marked as provisional unless reviewed; recorded review decisions override source labels. The private split manifest and review inventory are intentionally excluded from Git. [The aggregate audit summary](reports/data_summary.json) and the split-building code document the public, reproducible parts of the process. Results from older split versions remain in the JSON record with a historical status and should **not** be compared directly with current-split results.

## Models and findings

The initial CNN uses two convolution–ReLU–pooling blocks. A four-convolution variant extends its channels through `3→16→32→64→128`. With four `2×2` pooling operations, a `128×128` input reaches a `128×8×8` feature tensor before the classifier. The regularized CNN also uses horizontal flip during training, `Dropout(0.3)`, Adam weight decay of `1e-4`, and a `ReduceLROnPlateau` scheduler driven by validation loss.

ResNet18 uses ImageNet-compatible resizing and normalization. Feature extraction trains only the new eight-class head; fine-tuning then trains `layer4` and the head with separate learning rates (`1e-4` and `1e-3`, respectively). The final checkpoint was selected by **validation accuracy**, not by the loss-minimizing epoch.

| Model | Correct / 867 | Accuracy | Macro-F1 | Validation loss at selected checkpoint |
|---|---:|---:|---:|---:|
| Two-convolution CNN | 743 | 85.7% | 0.857 | 0.9704 |
| Four-convolution CNN, four MaxPool | 768 | 88.6% | 0.885 | 0.7434 |
| Four MaxPool + combined regularization | 788 | 90.9% | 0.908 | 0.4456 |
| Four AvgPool + the same regularization | 777 | 89.6% | 0.896 | 0.5062 |
| ResNet18 feature extraction | 797 | 91.9% | — | 0.2274 |
| **ResNet18 `layer4` fine-tuning, selected** | **833** | **96.1%** | **0.961** | **0.2145** |

The MaxPool-versus-AvgPool comparison holds the four-convolution architecture, parameter count (**162,984**), data split, initial weights, optimizer, regularization, and 30-epoch budget fixed. Four MaxPool gave **11 more correct predictions** than four AvgPool on this validation run. This is evidence for this setup, not a claim that MaxPool is universally better.

<details>
<summary>All 18 current-split CNN runs</summary>

`Val loss` is measured at the checkpoint selected by the **highest validation accuracy** for that run.

| Run | Train images | Best epoch | Correct / 867 | Accuracy | Macro-F1 | Val loss |
|---|---:|---:|---:|---:|---:|---:|
| Two-convolution CNN (30 epochs) | 3469 | 24/30 | 743 | 85.7% | 0.857 | 0.9704 |
| Two-convolution CNN (10 epochs) | 3469 | 10/10 | 737 | 85.0% | 0.849 | 0.7770 |
| Horizontal flip | 3469 | 23/30 | 736 | 84.9% | 0.848 | 0.9288 |
| Dropout 0.3 | 3469 | 11/30 | 734 | 84.7% | 0.845 | 0.7812 |
| Dropout 0.5 | 3469 | 15/30 | 737 | 85.0% | 0.850 | 0.7808 |
| AvgPool in two-convolution CNN | 3469 | 11/30 | 728 | 84.0% | 0.839 | 0.8831 |
| Weight decay 1e-4 | 3469 | 29/30 | 738 | 85.1% | 0.849 | 0.9757 |
| Learning-rate scheduler | 3469 | 24/30 | 743 | 85.7% | 0.857 | 0.7738 |
| Combined regularization, two-convolution CNN | 3469 | 14/30 | 740 | 85.4% | 0.851 | 0.6201 |
| Simulated imbalance, shuffled batches | 1512 | 24/30 | 654 | 75.4% | 0.734 | 1.4573 |
| Simulated imbalance, balanced batches | 1512 | 11/30 | 663 | 76.5% | 0.749 | 1.2097 |
| BCEWithLogitsLoss | 3469 | 14/30 | 741 | 85.5% | 0.854 | 0.2087 |
| Four convolutions, 32→32, two MaxPool | 3469 | 27/30 | 741 | 85.5% | 0.852 | 1.3276 |
| Four convolutions, 64→64, two MaxPool | 3469 | 26/30 | 743 | 85.7% | 0.853 | 1.4758 |
| Four convolutions, 64→128, two MaxPool | 3469 | 19/30 | 744 | 85.8% | 0.857 | 1.0625 |
| Four convolutions, 64→128, four MaxPool | 3469 | 15/30 | 768 | 88.6% | 0.885 | 0.7434 |
| Four MaxPool + combined regularization | 3469 | 20/30 | 788 | 90.9% | 0.908 | 0.4456 |
| Four AvgPool + combined regularization | 3469 | 28/30 | 777 | 89.6% | 0.896 | 0.5062 |

The 10-epoch baseline has a shorter training budget. The two simulated-imbalance runs use only **1,512 source-labelled training images** and should be compared with each other. BCE uses a different objective and its numeric loss scale is **not directly comparable** with CrossEntropy. All other rows use the same 867-image validation set; most use 30 epochs and 3,469 training images.

</details>

Other controlled comparisons include dropout (`p=0`, `0.3`, `0.5`), training-only augmentation, average versus max pooling, weight decay, a fixed learning rate versus a scheduler, and CrossEntropy versus one-hot `BCEWithLogitsLoss`. Under a reproducible simulated imbalance, balanced batches improved accuracy from **75.4% to 76.5%** relative to ordinary shuffled batches on the same 1,512-image training subset. These experiments and their per-epoch histories are recorded in [the experiment results](reports/experiment_results.json) and the notebooks.

## Error analysis and human review

The project records per-class precision, recall, F1, confusion counts, normalized confusion matrices, training curves, and inspected misclassifications. `kamyun` and `kamyunet` remain separate output classes after reviewing their mutual confusion and the intended label definitions. Uncertain and potentially mislabeled `unclean` images are treated explicitly rather than silently assumed to be clean.

The selected ResNet18 checkpoint uses a **0.90** softmax-confidence threshold chosen from internal validation. On those 867 validation images, **35** were sent to review, including **11** of the model's **34** mistakes. The other **23** mistakes were not caught by that threshold. The review flag therefore helps triage cases but is not a substitute for inspecting the prediction, especially under domain shift or on unrelated objects.

![CNN baseline training curves](reports/figures/cnn_baseline_curves.png)

![CNN baseline confusion matrices](reports/figures/cnn_baseline_confusion_matrices.png)

## Repository layout

```text
api.py                    FastAPI upload and health endpoints
predict.py                Reusable ResNet18 predictor and JSON CLI
scripts/data_split.py     Private, duplicate-aware development split
scripts/run_experiments.py CNN ablations and checkpoint/report writing
notebooks/01_data_audit.ipynb
notebooks/02_cnn_baseline.ipynb
reports/data_summary.json
reports/experiment_results.json
reports/figures/           Aggregate plots
requirements.txt          Recorded Python dependencies
```

The ignored `dataset/`, `dataset_extra/`, `checkpoints/`, and private review/split files are required only where the relevant training or inference step uses them.

## Setup and reproduction

The recorded environment used **Python 3.14** and the versions in [`requirements.txt`](requirements.txt). From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Full training reproduction requires authorized access to both image batches and the private review inventory. Place the datasets in `dataset/` and `dataset_extra/`, and keep the reviewed inventory at `reports/data_review_decisions.csv`. These paths are ignored by Git. Then create the private split:

```bash
python scripts/data_split.py --write
```

The split code checks exact duplicates, builds near-duplicate groups, and writes the ignored `reports/base_split.json`. Its fingerprint must match the one recorded with an experiment before results are compared. CNN experiments can be rerun independently, for example:

```bash
python scripts/run_experiments.py four_conv_128_4pool_regularized --epochs 30
python scripts/run_experiments.py four_conv_128_4avgpool_regularized --epochs 30
```

The data-audit workflow is in [`notebooks/01_data_audit.ipynb`](notebooks/01_data_audit.ipynb); the CNN, transfer-learning, and error-analysis workflow is in [`notebooks/02_cnn_baseline.ipynb`](notebooks/02_cnn_baseline.ipynb). A locally available final checkpoint is required for inference. Checkpoints are intentionally **not** distributed in this repository.

## Prediction interfaces

The command-line interface accepts one image path and writes JSON:

```bash
python predict.py /path/to/cropped_vehicle.jpg
```

Start the local FastAPI server from the repository root:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive upload form. In Postman, send **POST** `http://127.0.0.1:8000/predict`, choose **Body → form-data**, create a key named **`file`**, set its type to **File**, and select an image. The same request with `curl` is:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/cropped_vehicle.jpg"
```

`GET /health` reports model readiness and the eight class names. The server loads the final ResNet18 checkpoint once and reuses it across requests. It returns HTTP 400 for an unreadable image, 413 for an upload over 10 MB, and 503 if the checkpoint is unavailable. The response has this structure (illustrative values):

```json
{
  "predicted_class": "taxi",
  "confidence": 0.93,
  "probabilities": {
    "ambulance": 0.01,
    "autobus": 0.01,
    "kamyun": 0.01,
    "kamyunet": 0.01,
    "minibus": 0.01,
    "savari": 0.01,
    "taxi": 0.93,
    "vanet": 0.01
  },
  "needs_review": false
}
```

For a quick demo, use a publicly available **cropped vehicle** image rather than a private dataset image. Predictions on arbitrary internet photos illustrate the interface; they are not an accuracy evaluation.

## Repository safety

Raw datasets, private split/review files, and trained checkpoints are excluded by [`.gitignore`](.gitignore). The repository contains code and aggregate results, not the NDA-protected images or model weights. The reported scores are from **internal validation**; final performance on the held-out evaluator dataset remains unknown. The API can be checked manually through Postman or `/docs` with a cropped vehicle image that is not part of the private dataset.
