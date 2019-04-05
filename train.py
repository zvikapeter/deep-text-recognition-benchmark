import os
import sys
import time
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch_baidu_ctc import CTCLoss

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation


def train(opt):
    """ dataset preparation """
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    valid_dataset = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print('-' * 80)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt.imgH, opt.imgW, opt.input_channel, opt.output_channel, opt.hidden_size,
                  opt.num_class, opt.batch_max_length,
                  Transformation=opt.Transformation, FeatureExtraction=opt.FeatureExtraction,
                  SequenceModeling=opt.SequenceModeling, Prediction=opt.Prediction)
    print('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    if opt.continue_model != '':
        print(f'loading pretrained model from {opt.continue_model}')
        model.load_state_dict(torch.load(opt.continue_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = CTCLoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.continue_model != '':
        start_iter = int(opt.continue_model.split('_')[-1].split('.')[0])
        print(f'continue to train, start_iter: {start_iter}')

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    i = start_iter

    while(True):
        # train part
        for p in model.parameters():
            p.requires_grad = True

        cpu_images, cpu_texts = train_dataset.get_batch()
        image = cpu_images.cuda()
        text, length = converter.encode(cpu_texts)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, length, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format
            cost = criterion(preds, text, preds_size, length) / batch_size

        else:
            preds = model(image, length, text)
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if i % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            print(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
            with open(f'./saved_models/{opt.name}/log_train_data.txt', 'a') as log:
                log.write(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}\n')
            loss_avg.reset()

            model.eval()
            valid_loss, current_accuracy, current_norm_ED, preds, gts, infer_time = validation(
                model, criterion, valid_loader, converter, opt)
            model.train()

            # for log
            with open(f'./saved_models/{opt.name}/log_valid_data.txt', 'a') as log:
                for pred, gt in zip(preds[:5], gts[:5]):
                    if 'CTC' not in opt.Prediction:
                        pred = pred[:pred.find('[s]')]
                        gt = gt[:gt.find('[s]')]
                    print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
                    log.write(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')

                valid_log = f'[{i}/{opt.num_iter}] valid loss: {valid_loss:0.5f}'
                valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
                print(valid_log)
                log.write(valid_log + '\n')

            # keep best accuracy model
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), f'./saved_models/{opt.name}/best_accuracy.pth')
            if current_norm_ED < best_norm_ED:
                best_norm_ED = current_norm_ED
                torch.save(model.state_dict(), f'./saved_models/{opt.name}/best_norm_ED.pth')
            print(f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_norm_ED:0.2f}')

        # save model per 1e+5 iter.
        if (i + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.name:
        opt.name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.name += f'-Seed{opt.manualSeed}'
        # print(opt.name)

    os.makedirs(f'./saved_models/{opt.name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        opt.batch_size = opt.batch_size * opt.num_gpu
        opt.workers = opt.workers * opt.num_gpu
        print('multiplied batch_size', opt.batch_size)

    train(opt)