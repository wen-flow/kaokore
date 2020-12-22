conda: conda env create -f kaokore.yml
pip: pip install -r requirements.txt

support backbone:
'alexnet', 'googlenet', 'lenet', 'mobilenetv2', 
'resnet34', 'resnet101', 'vgg11', 'vgg13', 'vgg16',
'vgg19', 'mobilenetv3small'

change backbone in tools/opt.py

train:
cd path/to/kaokore
python train.py

predict:
cd path/to/kaokore
python predict.py

dataset:
链接: https://pan.baidu.com/s/1-VGQXj5e9iCfWeEBOzUvEQ  密码: 6uik
解压到datasets/
