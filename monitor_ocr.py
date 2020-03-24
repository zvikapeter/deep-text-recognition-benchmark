import string
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import cv2

from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_parameters():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='demo_image_monitor', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='PreTrained/TPS-ResNet-BiLSTM-Attn.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=5, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if opt.rgb:
        opt.input_channel = 3

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    return opt


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        # for index, l in enumerate(length):
        #     text = ''.join([self.character[i] for i in text_index[index,:]])
        #     texts.append(text)
        # return texts

        for text_raw in text_index:
            text = ''.join([self.character[i] for i in text_raw])
            texts.append(text)
        return texts


class ModelOCR(object):

    def __init__(self):

        # set parameters
        self.opt = set_parameters()

        # initialize text-label and text-index converter
        self.converter = AttnLabelConverter(self.opt.character)

        self.opt.num_class = len(self.converter.character)


    def load_model(self):

        opt = self.opt

        # load architecture
        model = Model(opt)

        # move to device
        model = torch.nn.DataParallel(model).to(device)

        # load weights
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        # set model mode to eval
        model.eval()

        self.model = model


    def preprocess_inputs(self, bbox_list, image):

        # crop bboxes from image
        images_list_raw = []
        for bb in bbox_list:
            images_list_raw.append(image[bb[1]:bb[3], bb[0]:bb[2]])

        images_list = []
        for single_roi in images_list_raw:
            images_list.append(cv2.resize(single_roi, (100, 32)))

        self.images_list = images_list

        return images_list


    def predict(self, images_list):

        opt = self.opt
        converter = self.converter
        model = self.model

        with torch.no_grad():

            batch_size = len(images_list)
            stackedImages = np.stack(images_list)
            stackedImages = np.expand_dims(stackedImages, axis=1)
            stackedImages = stackedImages / 255.
            torchStackedImages = torch.from_numpy(stackedImages[..., 0])
            torchStackedImages = torchStackedImages.type(torch.float32)
            torchStackedImages = torchStackedImages.sub_(0.5).div_(0.5)
            torchStackedImages = torchStackedImages.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            # predict
            preds = model(torchStackedImages, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)

            # convert indices to text
            preds_str = converter.decode(preds_index, length_for_pred)

            self.preds = preds
            self.preds_str = preds_str

            return preds, preds_str

    # def log_predictions(self, verbose=0):
    #
    #     log_file = self.log_file
    #     preds = self.preds
    #     preds_str = self.preds_str
    #
    #     log = open(log_file, 'a')
    #     dashed_line = '-' * 80
    #     head = f'{"predicted_labels":25s}\tconfidence score'
    #
    #     log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')


    def display(self, image_np, bbox_list, preds, preds_str, verbose=0):

        # preds = self.preds
        # preds_str = self.preds_str

        if verbose > 0:
            dashed_line = '-' * 80
            head = f'{"predicted_labels":25s}\tconfidence score'
            print(f'{dashed_line}\n{head}\n{dashed_line}')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        image_full = image_np.copy()
        imm = image_full.copy()

        madadim=[]
        for pred, pred_max_prob,rec in zip(preds_str, preds_max_prob, bbox_list):

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

            if verbose > 0:
                print(f'{pred:25s}\t{confidence_score:0.4f}')

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

        return imm



def main(bbox_list, image, save_image_path=None):

    model_ocr = ModelOCR()

    # load model
    model_ocr.load_model()

    # pre-process input
    images_list = model_ocr.preprocess_inputs(bbox_list=bbox_list, image=image)

    # predict
    preds, preds_str = model_ocr.predict(images_list)

    # display
    if save_image_path is not None:

        # add predicted text to image
        res = model_ocr.display(image, bbox_list, preds, preds_str, verbose=1)

        cv2.imwrite(save_image_path, res)

    return res

if __name__ == '__main__':


    # inputs
    bbox_list = np.load('demo_image_monitor/11_recs.npy')
    image = cv2.imread('demo_image_monitor/11.jpg')

    save_image_path = 'demo_image_monitor_out/res.jpg' # None

    # main
    res = main(bbox_list=bbox_list, image=image, save_image_path=save_image_path)

    print('Done!')

