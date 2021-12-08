import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from n3ml.model import DynamicModel_STBP_SNN


def validate(val_loader, model, encoder, criterion, opt):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            preds = model(encoder, images, opt.num_steps)
            labels_ = torch.zeros(torch.numel(labels), 10, device=labels.device)
            labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

            loss = criterion(preds, labels_)

            num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
            total_loss += loss.cpu().detach().numpy() * images.size(0)
            total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size)

    state_dict = torch.load(opt.pretrained)

    model = DynamicModel_STBP_SNN(batch_size=opt.batch_size)
    for m in state_dict['arch']:
        model.add_module(m[0], m[1])

    if torch.cuda.is_available():
        model.cuda()

    encoder = lambda x: (x > torch.rand(x.size(), device=x.device)).float()

    criterion = nn.MSELoss()

    acc, loss = validate(val_loader, model, encoder, criterion, opt)
    print("In test, loss: {} - acc: {}".format(loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_steps', default=15, type=int)
    parser.add_argument('--pretrained', default='pretrained/stbp_dynamic_acc_9897.pt')

    app(parser.parse_args())
