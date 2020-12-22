import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--model', default='mobilenetv3wen', type=str, help='alexnet, googlenet, lenet, mobilenetv2, resnet34, resnet101, vgg11, vgg13, vgg16, vgg19, mobilenetv3small, mobilenetv3wen')
    parser.add_argument('--dataset_dir', default='./datasets/kaokore', type=str, help='dataset dir')
    parser.add_argument('--end_epoch', default=200, type=int, help='Training ends at this epoch')
    parser.add_argument('--lr', default=0.0001, type=float, help='learn rate')
    parser.add_argument('--checkpoint', default=None, type=str, help='Continue training from pretrained (.pth)')
    parser.add_argument('--test_checkpoint', default='./weights/resnet34/best_model.pth', type=str, help='test weight')
    parser.add_argument('--val_per_epoch', default=1, type=int, help='eval per epoch')
    parser.add_argument('--batch_size', default=20, type=int, help='batch_size')
    parser.add_argument('--num_works', default=2, type=int, help='dataloader num_works')
    parser.add_argument('--model_save_dir', default='./weights/mobilenet', type=str, help='model_save_dir')
    parser.add_argument('--infer_out', default='./infer_out', type=str, help='infer_out_img')
    parser.add_argument('--onnx', default=True, type=bool, help='pytorch to onnx')
    args = parser.parse_args()
    return args