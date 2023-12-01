# Should you need to export Pytorch model with TorchScript to inference

Honestly, it took me a long time to learn about TorchScript, which is a new term for me at this point. But for an implemented model to be deployed for real-world applications across various platforms and devices, you need to know features, tools and libraries.


## **Table of contents**
- [1. Basics of Pytorch Model](#1)
- [2. What is TorchScript?](#2)
- [3. When, How they be used?](#3)


<a name="1"></a>

## 1. Basics of Pytorch Model
In Pytorch, `Module` is the basic unit of composition to you can define a simple Module. It contains:
+  A `__init__` function contains a set of parameters, sub-modules
+  A `forward` function. This is the code that is run when the module is invoked

A simple example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.linear1 = nn.Linear(320, 160)
        self.linear2 = nn.Linear(160, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.sigmoid(self.linear2(x))
        return x

classifier = Classification()
x = torch.rand((3, 320, 320))
print(classifier(x))
```

```bash
# Out:
tensor([[[0.4750, 0.4638],
         [0.4459, 0.4672],
         [0.4322, 0.4196],
         ...,
         [0.5129, 0.4328],
         [0.4635, 0.4232],
         [0.4701, 0.4578]],

        [[0.4487, 0.3940],
         [0.4664, 0.4431],
         [0.4613, 0.3938],
         ...,
         [0.4815, 0.4607],
         [0.4521, 0.4295],
         [0.4985, 0.4128]],

        [[0.4075, 0.4178],
         [0.4813, 0.4567],
         [0.4597, 0.3762],
         ...,
         [0.5193, 0.4664],
         [0.4907, 0.3778],
         [0.5108, 0.4237]]], grad_fn=<SigmoidBackward0>)
```

`Eager mode` is the default mode in Pytorch, which means that the code is executed immediately, as if it were a normal Python script. This allows for more intuitive and interactive coding, as you can see the results of each line of code and use print statements and breakpoints to debug.

`Graph mode` is the default model in Tensorflow, which means that the code is converted into a graph representation, and then executed by a runtime engine. This allows for better performance and scalability, as the graph can be optimized and parallelized across multiple devices.

<a name="2"></a>

## 2. What is TorchScript?
With TorchScript, Pytorch provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph model for speed, optimization and functionality in C++ runtime enviroments.

Pytorch provides two methods to convert `nn.Module` into a graph represented in TorchScript format: `trace` and `script`. 
+ `torch.jit.trace(model, input)`: TorchScript records Intermediate Representations(IR/Operations) as a graph.
+ `torch.jit.script(model)`: TorchScript records Intermediate Representations(IR/Operations) and Control Flow as a graph.

Let's begin an example with `trace`: 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.linear1 = nn.Linear(320, 160)
        self.linear2 = nn.Linear(160, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.sigmoid(self.linear2(x))
        return x

classifier = Classification()
x = torch.rand((3, 320, 320))
traced_classifer = torch.jit.trace(classifier, x)
print(traced_classifier)
# TorchScript records Intermediate Representations(IR/Operations) as a graph.
print(traced_classifier.graph)
# Use `.code` property to give Python-syntax interpretatin of the code.
print(traced_classifier.code)
```

Next, an example with `script`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputConfirm(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class Classification(nn.Module):
    def __init__(self, ic):
        super(Classification, self).__init__()
        self.ic = ic
        self.linear1 = nn.Linear(320, 160)
        self.linear2 = nn.Linear(160, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.sigmoid(self.linear2(x))
        return x

ic = InputConfirm()
classifier = Classification(ic)
x = torch.rand((3, 320, 320))
traced_ic = torch.jit.trace(ic, x)
traced_classifier = torch.jit.trace(classifier, x)
# Use .code property to give Python-syntax interpretatin of the code.
print(traced_ic.code)
print(traced_classifier.code)
```
```python
# OUT: print traced_ic.code
# <stdin>:3: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
def forward(self,
    x: Tensor) -> Tensor:
  return x

# OUT: print traced_classifier.code
def forward(self, x: Tensor) -> Tensor:
  linear2 = self.linear2
  linear1 = self.linear1
  _0 = (linear2).forward((linear1).forward(x, ), )
  return torch.sigmoid(_0)
```

Things we see: TorchScript does not record operation `if-else` (control flow) in `.code` output. TorchScript provides a script complier, 
which does direct analysis this problem with `torch.jit.scipt`:

```python
traced_ic = torch.jit.script(ic)
traced_classifier = torch.jit.scipt(classifier)
# Use .code property to give Python-syntax interpretatin of the code.
print(traced_ic.code)
print(traced_classifier.code)
```
<a name="3"></a>
## 3. When, How they be used?

As above mentioned, TorchScript provides ease-to-use and convert a runtime engine. This allows for better performance and scalability, as the graph can be optimized and parallelized across multiple devices.

Pytorch provides python's APIs to save and load TorchScript format to/from disk in an archive:

```python
traced_classifier.save('classifier.torchscript')
loaded_script = torch.jit.load('classifier.torchscript')
print(loaded_script)
print(loaded_script.code)
```

Futhermore, Pytorch also provides API to can be loaded and executed from C++, with no dependency on Python. Please see [LOADING A TORCHSCRIPT MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html) tutorial.

## Reference
+ [INTRODUCTION TO TORCHSCRIPT](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)
+ [TorchScript và PyTorch JIT](https://www.youtube.com/watch?v=2awmrMRf0dA)
