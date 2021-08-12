# SceneTransformer-pytorch
## Install dependencies
```pip install -r requirements.txt```
## Generate idx file for tfrecord dataset
```python datautil/create_idx.py```
## Predict model with SceneTransformer model (untrained)
```CUDA_VISIBLE_DEVICES=0 python tmp.py```