import torch
import torch.nn as nn
from torchvision.transforms import Normalize, ToTensor, Compose, CenterCrop, RandomHorizontalFlip,RandomResizedCrop, Resize
import os
import torch.optim as optim
from tensorboardX import SummaryWriter
from tools import dataloader_pytorch
from tools.utils import load_checkpoint, create_model
from tools.opt import parse_opts
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib as plt



opt = parse_opts()

def train(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": Compose([RandomResizedCrop(224),\
                          RandomHorizontalFlip(),\
                          ToTensor(),\
                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": Compose([Resize(256),\
                        CenterCrop(224),\
                        ToTensor(),\
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    assert os.path.exists(opt.dataset_dir), "{} path does not exist.".format(opt.dataset_dir)
    train_dataset = dataloader_pytorch.Kaokore(
        root=opt.dataset_dir, split="train", transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_works)
    val_dataset = dataloader_pytorch.Kaokore(
        root=opt.dataset_dir, split="dev", transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_works)
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    print("using {} images for training, {} images fot validation.".format(train_num, val_num))
    opt, net = create_model(opt)
    tf_run = os.path.join('./runs', opt.model)
    writer = SummaryWriter(tf_run)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr_list = list()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    start_epoch = 0
    net, optimizer, start_epoch = load_checkpoint(model=net, checkpoint_PATH=opt.checkpoint,optimizer=optimizer, epoch= start_epoch)
    best_acc = 0.0
    for epoch in range(start_epoch, opt.end_epoch):
        # train
        net.train()
        optimizer.step()
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        for step, data in enumerate(train_loader, start=0):
            images, labels, _ = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            # print train process
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f} epoch: {} lr: {}".format(int(rate*100), a, b, loss, epoch+1, optimizer.state_dict()['param_groups'][0]['lr']), end="")
        writer.add_scalar('train_loss', loss , epoch*train_num+opt.batch_size*len(data))
        print()
        if (epoch+1) % opt.val_per_epoch == 0:
            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            os.mkdir(opt.model_save_dir) if not os.path.exists(opt.model_save_dir) else print()
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels, _ = val_data
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    loss = loss_function(outputs, val_labels.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += (predict_y == val_labels.to(device)).sum().item()
                val_accurate = acc / val_num
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(), \
                                'optimizer': optimizer.state_dict()}, \
                               '{0}/best_model.pth'.format(opt.model_save_dir))
                    if opt.onnx:
                        dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
                        input_names = ["actual_input_1"]
                        output_names = ["output1"]
                        torch.onnx.export(net, dummy_input, '{0}/cls.onnx'.format(opt.model_save_dir), verbose=True,\
                                          input_names=input_names,
                                          output_names=output_names)
                print('val_loss: %.3f ,test_accuracy: %.3f' %(loss, val_accurate))
                writer.add_scalar('val_loss', loss, epoch+1)
                writer.add_scalar('val_accurate', val_accurate, epoch+1)
            net.train()
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(), \
                    'optimizer': optimizer.state_dict()}, \
                   '{0}/epoch_{1}.pth'.format(opt.model_save_dir, epoch + 1))

    print('Finished Training')
    plt.plot(range(100), lr_list, color='r')


if __name__ == '__main__':
    train(opt)
