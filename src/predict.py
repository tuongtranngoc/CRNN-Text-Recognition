from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from src.models.crnn import CRNN
from src.utils.logger import Logger
from src.data.transformation import TransformCRNN

logger = Logger.get_logger("PREDICTION")


class Predictor:
    def __init__(self, args) -> None:
        self.args = args
        __, self.id2char = self.map_char2id()
        self.model = CRNN(len(self.id2char)+1)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.to(self.args.device)
        self.model.eval()
        self.load_exported_model()
        self.transform = TransformCRNN()

    def load_exported_model(self):
        if self.args.export_format == 'torchscript':
            logger.info("Loading model for torchscript inference...")
            w = str(self.args.model_path).split('.pth')[0] + '.torchscript'
            self.ts = torch.jit.load(w, map_location=self.args.device)
            self.ts.float()

        if self.args.export_format == 'onnx':
            logger.info("Loading model for onnx inference...")
            w = str(self.args.model_path).split('.pth')[0] + '.onnx'
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(w, providers=providers)
            self.binding = self.session.io_binding()
            
        if self.args.export_format == 'tensorrt':
            logger.info("Loading model for tensorrt inference ...")
            w = str(self.args.model_path).split('.pth')[0] + '.engine'
            import tensorrt as trt
            from collections import OrderedDict, namedtuple
            logger_trt = trt.Logger(trt.Logger.INFO)
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            with open(w, 'rb') as f, trt.Runtime(logger_trt) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())  # read engine
            self.context = model.create_execution_context()
            self.bindings = OrderedDict()
            self.output_names = []
            
            for i in range(model.num_bindings):
                name = model.get_tensor_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if not model.binding_is_input(i):
                    self.output_names.append(name)
                shape = tuple(self.context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.args.device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

        if self.args.export_format == 'paddle':
            logger.info("Loading model for paddle to inference ...")
            w = str(self.args.model_path).split('.pth')[0] + '_paddle_model'
            import paddle.inference as pdi  # noqa
            w = Path(w)
            if not w.is_file():  # if not *.pdmodel
                w = next(w.rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            config = pdi.Config(str(w), str(w.with_suffix('.pdiparams')))
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            self.predictor = pdi.create_predictor(config)
            self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
            self.output_names = self.predictor.get_output_names()

    def preprocess(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = self.transform.transform(img)
            img = img.unsqueeze(0)
            return img
        else:
            Exception("Not exist image path")
    
    def map_char2id(self):
        dict_char2id = {}
        dict_id2char = {}
        with open(cfg['Global']['character_dict_path'],'r', encoding='utf-8') as f_dict:
            char_list = f_dict.readlines()
            for i, char in enumerate(char_list):
                char = char.strip('\n')
                if char not in dict_char2id:
                    dict_char2id[char] = i + 1
                    dict_id2char[i + 1] = char
        f_dict.close()
        return dict_char2id, dict_id2char
    
    def post_process(self, labels, blank=0):
        mapped_labels = []
        prev_label = None
        for l in labels:
            if l != prev_label:
                mapped_labels.append(l)
                prev_label = l
        mapped_labels = [l for l in mapped_labels if l != blank]
        return mapped_labels
    
    def post_decode(self, log_probs):
        log_prob = log_probs.cpu().detach().numpy()
        log_prob = log_prob.transpose((1, 0, 2))
        labels = np.argmax(log_prob[0], axis=-1)
        labels = self.post_process(labels)
        return labels
    
    def predict(self, img_path):
        img = self.preprocess(img_path).to(self.args.device)
        # inference with export format
        if self.args.export_format == 'torchscript':
            st = time.time()
            out = self.ts(img)
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")
            
        elif self.args.export_format == 'paddle':
            st = time.time()
            img = img.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(img)
            self.predictor.run()
            out = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")

        elif self.args.export_format == 'onnx':
            st = time.time()
            img = img.contiguous()
            self.binding.bind_input(
                name=self.session.get_inputs()[0].name,
                device_type=self.args.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(img.shape),
                buffer_ptr=img.data_ptr(),
            )
            out_shape = [26, 1, 37]
            out = torch.empty(out_shape, dtype=torch.float32, device='cuda:0').contiguous()
            self.binding.bind_output(
                name=self.session.get_outputs()[0].name,
                device_type=self.args.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(out.shape),
                buffer_ptr=out.data_ptr(),
            )
            self.session.run_with_iobinding(self.binding)
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")

        elif self.args.export_format == 'tensorrt':
            st = time.time()
            s = self.bindings['images'].shape
            assert img.shape == s
            self.binding_addrs['images'] = int(img.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            out = [self.bindings[x].data for x in sorted(self.output_names)]
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")

        else:
            st = time.time()
            out = self.model(img)
            logger.info(f"Runtime of Pytorch: {time.time()-st}")
        # convert out with export format
        if isinstance(out, list):
            out = out[-1]
        if isinstance(out, np.ndarray):
            out = torch.tensor(out)
        log_prob = F.log_softmax(out, dim=2)
        decoded_id = self.post_decode(log_prob)
        decoded_text = ''.join([self.id2char[_id] for _id in decoded_id])
        logger.info(f"Result of image {img_path}: {decoded_text}")
        return decoded_text

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file")
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default='cuda', help="device inference (cuda or cpu)")
    parser.add_argument("--export_format", type=str, default='pytorch', help="Exported format of model to inference")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    predictor = Predictor(args)
    for _ in range(100):
        predictor.predict(args.image_path)
    