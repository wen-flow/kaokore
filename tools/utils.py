import torch
from model import alexnet, googlenet, lenet, mobilenet, resnet, vgg, mobilenetv3_small, mobilenetv3_wen


def load_checkpoint(model=None, checkpoint_PATH=None, optimizer=None, epoch=None):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        print('loading checkpoint ',checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        optimizer.load_state_dict(model_CKPT['optimizer'])
        epoch = model_CKPT['epoch']
    return model, optimizer, epoch

def create_model(opt=None):
    assert opt.model in ['alexnet', 'googlenet', 'lenet', 'mobilenetv2', 'resnet34', 'resnet101', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'mobilenetv3small', 'mobilenetv3wen']
    if opt.model == 'mobilenetv2':
        model = mobilenet.MobileNetV2(num_classes=opt.n_classes, )
        print('net is mobilenetv2!')
        opt.model_save_dir='./weights/mobilenetv2'
    elif opt.model == 'alexnet':
        model = alexnet.AlexNet(num_classes=opt.n_classes, init_weights=True)
        print('net is alexnet!')
        opt.model_save_dir = './weights/alexnet'
    elif opt.model == 'googlenet':
        model = googlenet.GoogLeNet(num_classes=opt.n_classes, init_weights=True)
        print('net is googlenet!')
        opt.model_save_dir = './weights/googlenet'
    elif opt.model == 'lenet':
        model = lenet.LeNet(num_classes=opt.n_classes)
        print('net is lenet!')
        opt.model_save_dir = './weights/lenet'
    elif opt.model == 'resnet34':
        model = resnet.resnet34(num_classes=opt.n_classes)
        print('net is resnet34!')
        opt.model_save_dir = './weights/resnet34'
    elif opt.model == 'resnet101':
        model = resnet.resnet101(num_classes=opt.n_classes)
        print('net is resnet101!')
        opt.model_save_dir = './weights/resnet101'
    elif opt.model == 'vgg11':
        model = vgg.vgg(model_name="vgg11", num_classes=opt.n_classes, init_weights=True)
        print('net is vgg11!')
        opt.model_save_dir = './weights/vgg11'
    elif opt.model == 'vgg13':
        model = vgg.vgg(model_name="vgg13", num_classes=opt.n_classes, init_weights=True)
        print('net is vgg13!')
        opt.model_save_dir = './weights/vgg13'
    elif opt.model == 'vgg16':
        model = vgg.vgg(model_name="vgg16", num_classes=opt.n_classes, init_weights=True)
        print('net is vgg16!')
        opt.model_save_dir = './weights/vgg16'
    elif opt.model == 'vgg19':
        model = vgg.vgg(model_name="vgg19", num_classes=opt.n_classes, init_weights=True)
        print('net is vgg19!')
        opt.model_save_dir = './weights/vgg19'
    elif opt.model == 'mobilenetv3small':
        model = mobilenetv3_small.MobileNetV3_small(num_classes=opt.n_classes)
        print('net is mobilenetv3small!')
        opt.model_save_dir = './weights/mobilenetv3small'
    elif opt.model == 'mobilenetv3wen':
        model = mobilenetv3_wen.MobileNetV3_small(num_classes=opt.n_classes)
        print('net is mobilenetv3wen!')
        opt.model_save_dir = './weights/mobilenetv3wen'
    return opt, model



