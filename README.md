# SceneTransformer
This repo is unofficial pytorch implementation of SceneTransformer
## 1. Install dependencies
```pip install -r requirements.txt```
## 2. Put tfrecord data file at ./data/tfrecords

## 3. Generate idx file for tfrecord dataset
```python datautil/create_idx.py```
## 4. Predict model with SceneTransformer model (untrained)
```CUDA_VISIBLE_DEVICES=0 python tmp.py```


# Rasterized Waymo Data
## Follow waymo_rasterized_ex.ipynb
