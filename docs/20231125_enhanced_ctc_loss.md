# Enhanced CTC losss

In OCR recognition, CRNN is a text recognition algorithm as a starting point for improving cutting-edge text recognition algorithms at the moment. CTCLoss is an indispensable part of CRNN and is used to calculate network loss. In the post, I will introduce some bag of tricks for improving CTCLoss.

## 1. Focal-CTC Loss

For Focal Loss, it was mainly to solve the problem of a serious imbalance in the ratio of positive and negative samples in training (imbalance between foreground and background in object detection). Starting from the cross entropy (CE) loss for binary classification:

$$\text{CE}(p,y)=-\log(p) \space \text{if }y=1$$
$$\text{CE}(p,y)=-\log(1-p) \space \text{if otherwise}$$

Where $y \in \lbrace \pm 1 \rbrace$ specifies the ground-truth class and $p\in \lbrack 0, 1 \rbrack$ is the model's estimated probability for the class with label $y=1$.

TO reduce the weight of a large number of simple negative samples in training and also can be understood as a kind of difficult sample mining. The form of the Focal Loss is as below:

$$\text{FL}(p)=-\alpha(1-p)^{\gamma}*log(p)$$

where $\alpha$ - balanced variant of the focal loss, $(1-p)^{\gamma}$ - a modulating factor with tunable focusing parameter $\gamma \geq 0$.

For the original CTCLoss, The objective is to minimize the negative log-likelihood of conditional probability of ground truth

$$L_{CTC}=-log \space p(l_i|y_i)$$

where $l_i$ is ground-truth label sequence and $y_i$ is the sequence preoduced by input image. Applying the idea of Focal Loss to improve the accuracy of recognition. Therefor, Its definition is as follows:

$$L_{FocalCTCT}=-\alpha(1-p)^{\gamma}*log(p)$$

In the experiment, the value of $\gamma=2, \alpha=1$, see this for specific implementation: [losses.py](https://github.com/tuongtranngoc/CRNN-Text-Recognition/blob/main/src/utils/losses.py)
