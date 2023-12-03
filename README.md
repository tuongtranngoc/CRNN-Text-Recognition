# CRNN
An implementation of CRNN algorithm using Pytorch framework

The most typical CTC-algorithm is CRNN (Convolutional Recurrent Neural Network), which introduces the bidirectional LSTM (Long Short-Term Memory) to enhance the context modeling. And experiements have proved that the bidirectional LSTM module can effectively extract the contextual information of the picture, and finally enter the output feature sequence into the CTC module, and decode the sequence result.

<p>
    <image src="images/CRNN.png">
</p>

## Evironment

The dependences are listed in `requirements.txt`. Please install follow the command as bellow:
```bash
pip install -r requirements.txt
```

## Data Preparation
You can download and use [ICDAR2015]() or [MJSynth]() dataset for training model. 

After that,
+ Put the data folder under the `datasets` directory
+ Setup config file for each dataset in `src/__init__.py`: replace `CFG_PATH` augument by `src/config/rec_lmdb.yml` or `src/config/rec_icdar15.yml`

## Training
Before training, please modify configurations in `src/config/rec_lmdb.yml` or `src/config/rec_icdar15.yml`
```bash
python -m src.train
```
To train the model with pytorch lightning
```bash
python -m src.pl_modules.pl_train
```
## Evaluation
```bash
python -m src.eval
```

## Export model
```bash
python -m src.exports.exporter --export_format <torchscript/onnx/paddle/tensorrt> --device cuda
```

## Prediction
```bash
python -m src.predict --img_path <path_to_image_file> --export_format <torchscript/onnx/paddle/tensorrt> --device cuda
```
Example:

<p>
    <image src="images/tuongan.png">
</p>

```bash
python -m src.predict --image_path images/tuongan.png --export_format pt --device cuda
```





## Experiments

| Export format | inference time|
|---|---|
| Pytorch | 0.0008s |
| TorchScript | 0.0007s|
| ONNX | 0.001s|
| TensorRT | 0.0001s |
| Paddle | 0.01s|