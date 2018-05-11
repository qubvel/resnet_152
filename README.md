# ResNet-152 model for Keras

Extention of model taken [here](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6).  
Model made in Keras style with pretrained weights from ImageNet provided in release (load automatically during model initialization).  

### Example  
##### Inference example for this picture:  
<img src="https://github.com/qubvel/ResNet152/blob/master/imgs/cat.jpg" width="300" height="200">

##### Code
```python

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from model import ResNet152

# create model
model = ResNet152()

# define function for input preprocessing
def preprocess(x):
    x = resize(x, (224,224), mode='constant') * 255
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x

# prepare image
img = imread('./imgs/cat.jpg')
x = preprocess(img)

# make prediction and decode it
y = model.predict(x)
pred_title = decode_predictions(y, top=1)[0][0][1]

# print result
print(pred_title)
### tiget_cat
```
