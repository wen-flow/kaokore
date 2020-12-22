import torch
import os
from torchvision.transforms import Normalize, ToTensor, Compose, CenterCrop, RandomHorizontalFlip,RandomResizedCrop, Resize
from tools import dataloader_pytorch
from tools.opt import parse_opts
from tools.utils import create_model
from tqdm import tqdm
import cv2
import numpy as np
opt = parse_opts()

def infer(opt):
    cls = ['Man', 'Woman']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
            "test": Compose([Resize(256),\
                            CenterCrop(224),\
                            ToTensor(),\
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    assert os.path.exists(opt.dataset_dir), "{} path does not exist.".format(opt.dataset_dir)
    test_dataset = dataloader_pytorch.Kaokore(
            root=opt.dataset_dir, split="test", transform=data_transform['test'])
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_works)
    test_num = len(test_dataset)
    print("using {} images for testing.".format(test_num))
    opt, net = create_model(opt)
    net.to(device)

    model_CKPT = torch.load(opt.test_checkpoint)
    print('loading checkpoint ', opt.test_checkpoint)
    net.load_state_dict(model_CKPT['state_dict'])
    os.mkdir(opt.infer_out) if not os.path.exists(opt.infer_out) else print()
    with torch.no_grad():
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        img_root = os.path.join(opt.dataset_dir, 'images_256')
        for test_data in tqdm(test_loader):
            test_images, test_labels, image_filename = test_data
            image_dir = [os.path.join(img_root, img) for img in image_filename]
            outputs = net(test_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            for index, img_name in enumerate(image_dir):
                img = cv2.imread(img_name)
                img = cv2.putText(img, 'Class: ' + cls[np.array(predict_y.cpu())[index]], (5,250),\
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imwrite(os.path.join(opt.infer_out, img_name.split('/')[-1]), img)
            acc += (predict_y == test_labels.to(device)).sum().item()
        val_accurate = acc / test_num
        print('test_accuracy: %.3f' %(val_accurate))

if __name__ == '__main__':
    infer(opt)
