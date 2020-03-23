import string
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import cv2
from dataset import tensor2im
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
pilTrans = transforms.ToPILImage()
import mouse_crop

def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=640, imgW=352, keep_ratio_with_pad=opt.PAD)
    # AlignCollate_demo = AlignCollate(imgH=87, imgW=150, keep_ratio_with_pad=opt.PAD)

    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image_np = tensor2im(image_tensors[0])
            images_list, recs = mouse_crop.mouse_crop(image_np)
            # # pilTrans = transforms.ToPILImage()
            # # resizeTrans = transforms.Resize((opt.imgH, opt.imgW))
            # # image = resizeTrans(pilTrans(image))
            stackedImages = np.stack(images_list)
            stackedImages = np.expand_dims(stackedImages, axis=1)
            stackedImages= stackedImages/255.
            torchStackedImages = torch.from_numpy(stackedImages[...,0])
            torchStackedImages = torchStackedImages.type(torch.float32)
            torchStackedImages = torchStackedImages.sub_(0.5).div_(0.5)
            # torchStackedImages = torchStackedImages.expand(-1,1, -1, -1)
            torchStackedImages = torchStackedImages.to(device)

            # image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(torchStackedImages, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(torchStackedImages, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            image_full = cv2.imread(image_path_list[0])
            imm = image_full.copy()
            madadim=[]
            for pred, pred_max_prob,rec in zip(preds_str, preds_max_prob,recs):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.8
                    # Blue color in BGR
                    color = (0, 255, 0)
                    imm =cv2.rectangle(imm,(rec[0],rec[1]),(rec[2],rec[3]),(0,255,0),1)
                    imm = cv2.putText(imm, pred, (rec[0],rec[1]-5), font,
                                        fontScale, color, 2, cv2.LINE_AA)
                    madadim.append(pred)

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{image_path_list[0]:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{image_path_list[0]:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            head_text =''
            madadim[-2]=madadim[-2]+'/'+madadim[-1]
            madadim.pop(-1)
            madadim_name =['BPM', 'Respiration','SpO2','Bipm']
            y0, dy = 50,20
            fontScale = 0.5
            color = (255, 0, 0)
            for i,m in enumerate(madadim):
                y = y0 + i * dy
                imm = cv2.putText(imm, '{} : {}'.format(madadim_name[i],madadim[i]), (20, y), font,
                                  fontScale, color, 1, cv2.LINE_AA)

            cv2.imwrite(image_path_list[0].replace('demo_image_monitor','demo_image_monitor_out'), imm)
            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=5, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
