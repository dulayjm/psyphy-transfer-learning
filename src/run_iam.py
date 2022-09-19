import editdistance

def error_cer(r, h):
    # Remove any double or trailing
    r = u' '.join(r.split())
    h = u' '.join(h.split())

    return error_err(r, h)


def error_err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))


def error_wer(r, h):
    r = r.split()
    h = h.split()

    return err(r, h)


# In[92]:


import sys
import json
import os
from collections import defaultdict


def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']


# if __name__ == "__main__":
#     character_set_path = sys.argv[-1]
#     out_char_to_idx = {}
#     out_idx_to_char = {}
#     char_freq = defaultdict(int)
#     for i in range(1, len(sys.argv)-1):
#         data_file = sys.argv[i]
#         with open(data_file) as f:
#             data = json.load(f)

#         cnt = 1 # this is important that this starts at 1 not 0
#         for data_item in data:
#             for c in data_item.get('gt', ""):
#                 if c not in out_char_to_idx:
#                     out_char_to_idx[c] = cnt
#                     out_idx_to_char[cnt] = c
#                     cnt += 1
#                 char_freq[c] += 1

#     out_char_to_idx2 = {}
#     out_idx_to_char2 = {}

#     for i, c in enumerate(sorted(out_char_to_idx.keys())):
#         out_char_to_idx2[c] = i+1
#         out_idx_to_char2[i+1] = c

#     output_data = {
#         "char_to_idx": out_char_to_idx2,
#         "idx_to_char": out_idx_to_char2
#     }

#     for k,v in sorted(char_freq.iteritems(), key=lambda x: x[1]):
#         print(k, v)

#     print("Size:", len(output_data['char_to_idx']))

#     with open(character_set_path, 'w') as outfile:
#         json.dump(output_data, outfile)


# In[93]:


import numpy as np


def str2label(value, characterToIndex={}, unknown_index=None):
    if unknown_index is None:
        unknown_index = len(characterToIndex)

    label = []
    for v in value:
        # print(v)
        if v not in characterToIndex:
            continue
        label.append(characterToIndex[v])
    return np.array(label, np.uint32)


def label2input(value, num_of_inputs, char_break_interval):
    idx1 = len(value) * (char_break_interval + 1) + char_break_interval
    idx2 = num_of_inputs + 1
    print(idx1)
    print(idx2)
    input_data = [[0 for i in range(idx2)] for j in range(idx1)]

    cnt = 0
    for i in range(char_break_interval):
        input_data[cnt][idx2 - 1] = 1
        cnt += 1

    for i in range(len(value)):
        if value[i] == 0:
            input_data[cnt][idx2 - 1] = 1
        else:
            input_data[cnt][value[i] - 1] = 1
        cnt += 1

        for i in range(char_break_interval):
            input_data[cnt][idx2 - 1] = 1
            cnt += 1

    return np.array(input_data)


def label2str(label, indexToCharacter, asRaw, spaceChar="~"):
    string = u""
    for i in range(len(label)):
        if label[i] == 0:
            if asRaw:
                string += spaceChar
            else:
                break
        else:
            val = label[i]
            string += indexToCharacter[val]
    return string


def naive_decode(output):
    rawPredData = np.argmax(output, axis=1)
    predData = []
    for i in range(len(output)):
        if rawPredData[i] != 0 and not (i > 0 and rawPredData[i] == rawPredData[i - 1]):
            predData.append(rawPredData[i])
    return predData, list(rawPredData)


# In[94]:


# import json

# import torch
# from torch.utils.data import Dataset
# from torch.autograd import Variable

# from collections import defaultdict
# import os
# # import cv2
# from PIL import Image

# import numpy as np
# # fuck is this even cht ecode


# import random


# PADDING_CONSTANT = 255

# coeff = 1000

# def batch_collate(batch):
#     global coeff
#     batch = [b for b in batch if b is not None]
# #     #These all should be the same size or error


# #     print('the entire fucking batch is ', batch)
# #     print('len of the images array in the batch is', len(batch[0]))
# #     1/0


# #     print('in collate, the batch shape is ', len(batch))
# #     print('in collate, the batch 0 is ', len(batch[0]))
# #     print('in collate, the batch 0 is ', batch[0])

# #     print('type of bach is ', type(batch))
# #     print('type of bach is ', type(batch[0]))
# #     print(batch)


# #     print([b['line_img'].shape[0] for b in batch])

# #     print([b['line_img'].shape[2] for b in batch])


# #     print('batch shape 2 is', batch.shape[2])

# #
#     # are rts's just a directory ?
# #     1/0/
# #     assert len(set([b['line_img'].shape[0] for b in batch])) == 1
# #     assert len(set([b['line_img'].shape[2] for b in batch])) == 1

#     dim0 = batch[0]['line_img'].shape[0]
#     dim1 = max([b['line_img'].shape[1] for b in batch])
#     dim1 = dim1 + (dim0 - (dim1 % dim0))
#     # so it messes up here?
#     print('the object shape we are dealing with is', batch[0]['line_img'].shape)

#     dim2 = batch[0]['line_img'].shape[2]

#     all_labels = []
#     label_lengths = []
#     psychs = []


#     input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
#     for i in range(len(batch)):
#         b_img = batch[i]['line_img']
#         input_batch[i,:,:b_img.shape[1],:] = b_img
#         l = batch[i]['gt_label']
#         psych = batch[i]['psych']
#         all_labels.append(l)
#         label_lengths.append(len(l))
#         if psych is not None:
#             # print(psych)
#             # print(((200-psych)/len(l)))
#             # print("-----------")
#             psych = 200-psych
#             if psych < 0:
#                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#                 print((psych) / len(l))
#             psychs.append((psych/len(l))*coeff)
#         else:
#             psychs.append(0)
#     all_labels = np.concatenate(all_labels)
#     label_lengths = np.array(label_lengths)

#     line_imgs = input_batch.transpose([0,3,1,2])
#     line_imgs = torch.from_numpy(line_imgs)
#     labels = torch.from_numpy(all_labels.astype(np.int32))
#     label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

#     return {
#         "line_imgs": line_imgs,
#         "labels": labels,
#         "psychs": psychs,
#         "label_lengths": label_lengths,
#         "gt": [b['gt'] for b in batch]
#     }

# class HwDataset(Dataset):
#     def __init__(self,
#                  json_path,
#                  char_to_idx,
#                  img_height=32,
#                  root_path=".",
#                  augmentation=False,
#                  psychPath="./data/", # needs to exist you fuck ass
#                  randomW=False,
#                  coef=1000):

#         global coeff
#         with open(json_path) as f:
#             data = json.load(f)

#             # but we now need to loop through alll the directories
#             # so load root_dir (data) her


#         self.root_path = root_path
#         self.img_height = img_height
#         self.char_to_idx = char_to_idx
#         self.data = data
#         self.psychPath = psychPath
#         self.augmentation = augmentation
#         self.randomWeights = None
#         coeff = coef
#         if randomW:
#             self.randomWeights = np.random.randint(50, 200, len(self.data))


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):


#         # and on __getitem__
#         # do the json loading. I think.

#         item = self.data[idx]
#         # just worry about resizing after
#         # i think those cv2 augmentations just make it a square ...
#         img = np.array(Image.open(os.path.join(self.root_path, item['image_path'])))

# #         img = cv2.imread(os.path.join(self.root_path, item['image_path']))

#         if self.randomWeights is None:
#             try:
#                 # replace "magic number path"
#                 # print(item['image_path'].split('/')[-1].split('.')[0])
#                 mangled_string = self.psychPath+item['image_path'].split('/')[-1].split('.')[0]+"Time.json"
#                 print('mangled string is', mangled_string)

#                 with open(self.psychPath+item['image_path'].split('/')[-1].split('.')[0]+"Time.json") as f:
#                     psych = json.load(f)


#                 print("psych is after that loaddddd", psych)
#             except:
#                 psych = None
#         else:
#             print("WARNING!!! RANDOM PSYCHOMETRIC WEIGHTS ARE BEING USED")
#             psych = self.randomWeights[idx]
#             # print psych

#         if img is None:
#             print("Warning: image is None:", os.path.join(self.root_path, item['image_path']))
#             return None

# #         percent = float(self.img_height) / img.shape[0]
# #         print('percent is', percent)


# #         img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

# #         if self.augmentation:
# #             img = grid_distortion.warp_image(img, h_mesh_std=5, w_mesh_std=10)

#         img = img.astype(np.float32)
#         img = img / 128.0 - 1.0

#         gt = item['gt']
#         gt_label = str2label(gt, self.char_to_idx)

#         print('fuckign here')
#         print(type(img))
#         print(type(gt_label))
#         print(type(psych))
#         print(type(gt))
#         print("FLKSHJLKDFJSDLFJLFSDKJLKDSF")
#         print(img)
#         print(gt_label)
#         print(psych)
#         print(gt)


# #         1/0

#         # whyyyy is this reutrned as a list

#         ret_dict = {
#             "line_img": img,
#             "gt_label": gt_label,
#             "psych": psych,
#             "gt": gt,
#         }

#         print('the type of fucking ret_dict is', type(ret_dict))

#         return ret_dict


# In[95]:


# fresh copy of the code

import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np

import random
import string_utils

# import grid_distortiongrid_distortion

PADDING_CONSTANT = 255

coeff = 1000


def batch_collate(batch):
    global coeff
    batch = [b for b in batch if b is not None]
    # These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim1 = dim1 + (dim0 - (dim1 % dim0))
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []
    psychs = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i, :, :b_img.shape[1], :] = b_img
        l = batch[i]['gt_label']
        psych = batch[i]['psych']
        all_labels.append(l)
        label_lengths.append(len(l))
        if psych is not None:
            # print(psych)
            # print(((200-psych)/len(l)))
            # print("-----------")
            psych = 200 - psych
            if psych < 0:
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print((psych) / len(l))
            psychs.append((psych / len(l)) * coeff)
        else:
            psychs.append(0)
    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0, 3, 1, 2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    # psychs needs to be a tensor?

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "psychs": psychs,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }


class HwDataset(Dataset):
    def __init__(self, json_path, char_to_idx, img_height=32, root_path=".", augmentation=False, psychPath="",
                 randomW=False, coef=1000):
        global coeff
        with open(json_path) as f:
            data = json.load(f)
        self.root_path = root_path
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.data = data
        self.psychPath = psychPath
        self.augmentation = augmentation
        self.randomWeights = None
        coeff = coef
        if randomW:
            self.randomWeights = np.random.randint(50, 200, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(os.path.join(self.root_path, item['image_path']))
        if self.randomWeights is None:
            try:
                # replace "magic number path"
                # print(item['image_path'].split('/')[-1].split('.')[0])
                with open(self.psychPath + item['image_path'].split('/')[-1].split('.')[0] + "Time.json") as f:
                    psych = json.load(f)
            except:
                psych = None
        else:
            print("WARNING!!! RANDOM PSYCHOMETRIC WEIGHTS ARE BEING USED")
            psych = self.randomWeights[idx]
            # print psych

        if img is None:
            print("Warning: image is None:", os.path.join(self.root_path, item['image_path']))
            return None

        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        # no need to augment crazy

        #         if self.augmentation:
        #             img = warp_image(img, h_mesh_std=5, w_mesh_std=10)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = item['gt']
        gt_label = str2label(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt_label": gt_label,
            "psych": psych,
            "gt": gt
        }


# In[ ]:


# In[96]:


import torch
from torch import nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.reshape(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.reshape(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        return output


def create_model(config):
    crnn = CRNN(config['cnn_out_size'], config['num_of_channels'], config['num_of_outputs'], 512)
    return crnn


# In[97]:


# class MyResnet(nn.Module):
#     def __init__():
#         super.__init__(MyResnet,self)

#         self.backbone = resnet50(pretrained=True).to(device)
#         self.backbone.train()


# In[98]:


# so now we incorporate our other code into this one w/ the data uploaded

# use their train routine, but with your model training script stuff


# In[99]:


# their training routine
import json
# import character_set
import sys
# import crnn
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
# import error_rates
# import string_utils

from torchvision.models import resnet50

dtype = torch.cuda.FloatTensor

# their config
config = {
    "training_set_path": "/afs/crc.nd.edu/group/cvrl/scratch_31/sgrieggs/IAM_aachen/train.json",
    #     "training_set_path": "prepare_font_data/training.json",
    "validation_set_path": "/afs/crc.nd.edu/group/cvrl/scratch_31/sgrieggs/IAM_aachen/val.json",
    "image_root_directory": "/afs/crc.nd.edu/group/cvrl/scratch_31/sgrieggs/IAM_aachen/",
    "model_save_path": "sam_data.pt",
    "network": {
        "input_height": 60,
        "cnn_out_size": 1024,
        "learning_rate": 1e-4
    },
    "character_set_path": "char_set.json"
}
# but something to do with this later um

# # but what character set path to use
# idx_to_char, char_to_idx = load_char_set(config['character_set_path'])

# # we don't want that font traing stuff
# # we want the data that's actually in our path
# # but our labels are a set form a char file
# # and our rts are from the list of rt files per label, no?


# train_dataset = HwDataset(config['training_set_path'], char_to_idx, img_height=config['network']['input_height'], root_path=config['image_root_directory'], augmentation=True)
# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=batch_collate)

# test_dataset = HwDataset(config['validation_set_path'], char_to_idx, img_height=config['network']['input_height'], root_path=config['image_root_directory'])
# test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=batch_collate)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('device is', device)

# model = resnet50(pretrained=True).to(device)
# model.fc = nn.Linear(2048, 295)
# model.train()

# # model = create_model({
# #     'cnn_out_size': config['network']['cnn_out_size'],
# #     'num_of_channels': 3,
# #     'num_of_outputs': len(idx_to_char)+1
# # })

# if torch.cuda.is_available():
#     model.to(device)
#     dtype = torch.cuda.FloatTensor
#     print("Using GPU")
# else:
#     dtype = torch.FloatTensor
#     print("No GPU detected")

# optimizer = torch.optim.Adam(model.parameters(), lr=config['network']['learning_rate'])
# criterion = torch.nn.CrossEntropyLoss()
# lowest_loss = float('inf')


# print('sanity  is', train_dataloader)
# print(len(train_dataloader))
# print(len(train_dataset))

# # print('model is ', model)


# # dataiter = iter(train_dataloader)
# # batch = next(dataiter)


# # 1/0
# #
# for epoch in range(1000):
#     sum_loss = 0.0
#     steps = 0.0
#     for x in train_dataloader:
#         # then we care about how their dataset __getitem__
# #         return {
# #             "line_imgs": line_imgs,
# #             "labels": labels,
# #             "psychs": psychs,
# #             "label_lengths": label_lengths,
# #             "gt": [b['gt'] for b in batch]
# #         }

#         # so we could do some dataloader stuff, but this is what we need to change here
#         # psychs needs to be a torch tensor from the batch stuff

#         psychs = x['psychs']
#         if isinstance(psychs, list):
#             psychs = torch.LongTensor(psychs)
#         psychs = Variable(psychs, requires_grad=False)

# #         labels = torch.FloatTensor(labels)
# #         print('gt should be', x['gt'])

# #         gt = x['gt']
# #         if isinstance(gt, list):
# #             gt = torch.tensor(gt)
# #         gt = Variable(gt, requires_grad=False)

#         line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
#         labels =  Variable(x['labels'], requires_grad=False)
#         label_lengths = Variable(x['label_lengths'], requires_grad=False)

#         preds = model(line_imgs)
#         preds_size = Variable(torch.LongTensor([preds.size(0)] * preds.size(1)))
#         preds = preds.permute(1,0)

#         print('preds shape is', preds.shape)

# #         output_batch = preds.permute(1,0,2)
# #         out = output_batch.data.cpu().numpy()

# #         loss = criterion(preds, labels, preds_size, label_lengths)
#         labels = torch.LongTensor(labels)
#         labels = labels.to(device)

#         print('here')
#         print(line_imgs.shape)
#         print(preds.dtype)
#         print(labels.dtype)

#         loss = criterion(preds, labels)

#         # or
# #         loss = RtPsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         #if i == 0:
#         #    for i in xrange(out.shape[0]):
#         #        pred, pred_raw = string_utils.naive_decode(out[i,...])
#         #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
#         #        print(pred_str)

#         for j in range(out.shape[0]):
#             logits = out[j,...]
#             pred, raw_pred = naive_decode(logits)
#             pred_str = label2str(pred, idx_to_char, False)
#             gt_str = x['gt'][j]
#             cer = error_rates.cer(gt_str, pred_str)
#             sum_loss += cer
#             steps += 1

#     print("Training CER", sum_loss / steps)

#     sum_loss = 0.0
#     steps = 0.0
#     model.eval()
#     for x in test_dataloader:
#         line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
#         labels =  Variable(x['labels'], requires_grad=False, volatile=True)
#         label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)

#         preds = model(line_imgs).cpu()

#         output_batch = preds.permute(1,0,2)
#         out = output_batch.data.cpu().numpy()

#         for i, gt_line in enumerate(x['gt']):
#             logits = out[i,...]
#             pred, raw_pred = naive_decode(logits)
#             pred_str = label2str(pred, idx_to_char, False)
#             cer_ = cer(gt_line, pred_str)
#             sum_loss += cer_
#             steps += 1

#     print("Test CER", sum_loss / steps)

#     if lowest_loss > sum_loss/steps:
#         lowest_loss = sum_loss/steps
#         print("Saving Best")
#         dirname = os.path.dirname(config['model_save_path'])
#         if len(dirname) > 0 and not os.path.exists(dirname):
#             os.makedirs(dirname)

#         torch.save(model.state_dict(), os.path.join(config['model_save_path']))


# In[100]:


# import character_set
# import sys
# # import hwpsych_dataset
# # from hwpsych_dataset import HwDataset
# # import urnn, urnn2, urnn_window
# # import crnn, crnn2
# # import unet_hwr as unet
# import os
# import torch
# from torch.utils import data
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# # import error_rates
# # import string_utils
# import time
# from  torch.nn.modules.loss import CTCLoss
# from tqdm import tqdm

# psychPath = "./data/"


# class PsychCTC(torch.nn.Module):
#     def __init__(self,idx_to_char, char_to_idx, verbose = False):
#         super(PsychCTC, self).__init__()
#         self.citerion = CTCLoss(reduction = 'none', zero_infinity=True)
#         self.idx_to_char = idx_to_char
#         self.char_to_idx = char_to_idx
#         self.verbose = verbose


#     def forward(self, preds, labels, preds_size, label_lengths, psych):
#         # print(len(self.char_to_idx))
#         # might need preds_size to be 8
#         print('in the loss')
#         print('preds', preds.shape)
#         print('labels', labels.shape)


#         print('pred_size', preds_size.shape)
#         print('label_lenghts', label_lengths.shape)

#         loss = self.citerion(preds, labels, preds_size, label_lengths).cuda()
#         index = 0
#         # lbl = labels.detach().cpu().numpy()
#         lbl = labels.data.cpu().numpy()
#         lbl_len = label_lengths.data.cpu().numpy()
#         output_batch = preds.permute(1, 0, 2)
#         # out = output_batch.detach().cpu().numpy()
#         out = output_batch.data.cpu().numpy()
#         cer = torch.zeros(loss.shape)
#         for j in range(out.shape[0]):
#             logits = out[j, ...]
#             pred, raw_pred = naive_decode(logits)
#             pred_str = label2str(pred, self.idx_to_char, False)
#             gt_str = label2str(lbl[index:lbl_len[j] + index], self.idx_to_char, False)
#             index += lbl_len[j]
#             cer[j] = error_rates.cer(gt_str, pred_str)
#             # if self.verbose or psych[j] > 6000:
#             #     print(psych[j])
#         cer = Variable(cer, requires_grad = True).cuda()
#         loss = loss + (psych * cer)
#         return torch.sum(loss)


# # config_path = sys.argv[1]
# # try:
# #     jobID = sys.argv[2]
# # except:
# #     jobID = ""
# # print(jobID)

# # with open(config_path) as f:
# #     config = json.load(f)

# # try:
# #     model_save_path = sys.argv[3]
# #     if model_save_path[-1] != os.path.sep:
# #         model_save_path = model_save_path + os.path.sep
# # except:

# model_save_path = config['model_save_path']


# verbose = False

# dirname = os.path.dirname(model_save_path)
# print(dirname)
# if len(dirname) > 0 and not os.path.exists(dirname):
#     os.makedirs(dirname)

# # with open(config_path) as f:
# #     paramList = f.readlines()

# # for x in paramList:
# #     print(x[:-1])

# # baseMessage = ""

# # for line in paramList:
# #     baseMessage = baseMessage + line


# # print(baseMessage)

# idx_to_char, char_to_idx = load_char_set(config['character_set_path'])

# train_dataset = HwDataset(config['training_set_path'],
#                           char_to_idx,
#                           img_height=config['network']['input_height'],
#                           root_path=config['image_root_directory'],
#                           augmentation=False,
#                           psychPath=psychPath,
#                           randomW=False)

# try:
#     test_dataset = HwDataset(config['validation_set_path'],
#                              char_to_idx,
#                              img_height=config['network']['input_height'],
#                              root_path=config['image_root_directory'],
#                              psychPath=psychPath)
# except KeyError as e:
#     print("No validation set found, generating one")
#     master = train_dataset
#     print("Total of " +str(len(master)) +" Training Examples")
#     n = len(master)  # how many total elements you have
#     n_test = int(n * .1)
#     n_train = n - n_test
#     idx = list(range(n))  # indices to all elements
#     train_idx = idx[:n_train]
#     test_idx = idx[n_train:]
#     test_dataset = data.Subset(master, test_idx)
#     train_dataset = data.Subset(master, train_idx)

# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1,
#                               collate_fn=batch_collate)
# test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1,
#                              collate_fn=batch_collate)
# print("Train Dataset Length: " + str(len(train_dataset)))
# print("Test Dataset Length: " + str(len(test_dataset)))


# # if config['model'] == "crnn":
# #     print("Using CRNN")
# #     hw = crnn.create_model({
# #         'cnn_out_size': config['network']['cnn_out_size'],
# #         'num_of_channels': 3,
# #         'num_of_outputs': len(idx_to_char) + 1
# #     })
# # if config['model'].upper() == "UNET":
# #     print("Using UNET")
# #     hw = unet.create_model({
# #         'input_height': config['network']['input_height'],
# #         'cnn_out_size': config['network']['cnn_out_size'],
# #         'num_of_channels': 3,
# #         'num_of_outputs': len(idx_to_char) + 1,
# #         'bridge_width': config['network']['bridge_width']
# #     })
# # elif config['model'] == "urnn":
# #     print("Using URNN")
# #     hw = urnn.create_model({
# #         'input_height': config['network']['input_height'],
# #         'cnn_out_size': config['network']['cnn_out_size'],
# #         'num_of_channels': 3,
# #         'num_of_outputs': len(idx_to_char)+1,
# #         'bridge_width': config['network']['bridge_width']
# #     })
# # elif config['model'] == "urnn2":
# #     print("Using URNN with Curtis's recurrence")
# #     hw = urnn2.create_model({
# #         'input_height': config['network']['input_height'],
# #         'cnn_out_size': config['network']['cnn_out_size'],
# #         'num_of_channels': 3,
# #         'num_of_outputs': len(idx_to_char) + 1,
# #         'bridge_width': config['network']['bridge_width']
# #     })
# # elif config['model'] == "crnn2":
# #     print("Using original CRNN")
# #     hw = crnn2.create_model({
# #         'cnn_out_size': config['network']['cnn_out_size'],
# #         'num_of_channels': 3,
# #         'num_of_outputs': len(idx_to_char) + 1
# #     })
# # elif config['model'] == "urnn3":
# #     print("Using windowed URNN with Curtis's recurrence")
# #     hw = urnn_window.create_model({
# #         'input_height': config['network']['input_height'],
# #         'cnn_out_size': config['network']['cnn_out_size'],
# #         'num_of_channels': 3,
# #         'num_of_outputs': len(idx_to_char) + 1,
# #         'bridge_width': config['network']['bridge_width']
# #     })


# hw = create_model({
#         'cnn_out_size': config['network']['cnn_out_size'],
#         'num_of_channels': 3,
#         'num_of_outputs': len(idx_to_char) + 1
#     })

# resume = True
# try:
#     config['model_load_path']
# except KeyError:
#     resume = False

# pretrain = True
# try:
#     config['pretrained_load_path']
# except KeyError:
#     pretrain = False

# freeze = True
# try:
#     config['freeze_pretrained_weights']
# except KeyError:
#     freeze = False

# print("Freeze: " + str(freeze))
# metric = "CER"
# try:
#     metric = config['metric'].upper()
# except KeyError:
#     print("No metric listed, defaulting to Character Error Rate")
# print("Metric: " + metric)


# if resume:
#     hw.load_state_dict(torch.load(config['model_load_path']))
# elif pretrain:
#     print("loading pretrained weights")
#     model_param = hw.state_dict()
#     unet_param = torch.load(config['pretrained_load_path'])
#     for i in unet_param.keys():
#         model_param['UNet.' + i] = unet_param[i]
#     hw.load_state_dict(model_param)

# if freeze:
#     for param in hw.UNet.parameters():
#         param.requires_grad = False

# if torch.cuda.is_available():
#     hw.cuda()
#     dtype = torch.cuda.FloatTensor
#     print("Using GPU")
# else:
#     dtype = torch.FloatTensor
#     print("No GPU detected")

# optimizer = torch.optim.Adadelta(hw.parameters(), lr=config['network']['learning_rate'])
# criterion = PsychCTC(idx_to_char, char_to_idx, verbose=False)
# # criterion = CTCLoss(reduction='sum',zero_infinity=True)
# lowest_loss = float('inf')
# best_distance = 0
# for epoch in range(1000):
#     torch.enable_grad()
#     startTime = time.time()
#     message = "foo"
#     sum_loss = 0.0
#     sum_wer_loss = 0.0
#     steps = 0.0
#     hw.train()
#     disp_ctc_loss = 0.0
#     disp_loss = 0.0
#     gt = ""
#     ot = ""
#     loss = 0.0
#     # train_dataloader = torch.utils.data.random_split(train_dataloader, [16, len(train_dataloader) - 16])[0]
#     print("Train Set Size = " + str(len(train_dataloader)))
#     prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
#     for i, x in prog_bar:
#         # message = str("CER: " + str(disp_loss) +"\nGT: " +gt +"\nex: "+out+"\nProgress")
#         prog_bar.set_description(f'CER: {disp_loss} CTC: {loss} Ground Truth: |{gt}| Network Output: |{ot}|')
#         line_imgs = x['line_imgs']
#         psych = Variable(torch.Tensor(x['psychs']).type(dtype), requires_grad=True)
#         rem = line_imgs.shape[3] % 32
#         if rem != 0:
#             imgshape = line_imgs.shape
#             temp = torch.zeros(imgshape[0], imgshape[1], imgshape[2], imgshape[3] + (32 - rem))
#             temp[:, :, :, :imgshape[3]] = line_imgs
#             line_imgs = temp
#             del temp
#         line_imgs = Variable(line_imgs.type(dtype), requires_grad=False)

#         labels = Variable(x['labels'], requires_grad=False)
#         label_lengths = Variable(x['label_lengths'], requires_grad=False)

#         preds = hw(line_imgs).cpu()
#         preds_size = Variable(torch.IntTensor([preds.size(1)] * preds.size(0)))

#         print('pred size is ', preds_size.shape)

#         # output_batch = preds.permute(1,0,2)
#         out = preds.data.cpu().numpy()
#         preds = preds.permute(1, 0, 2)
#         loss = criterion(preds, labels, preds_size, label_lengths, psych)
#         # print(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # if i == 0:
#         #    for i in range(out.shape[0]):
#         #        pred, pred_raw = string_utils.naive_decode(out[i,...])
#         #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
#         #        print(pred_str)

#         for j in range(out.shape[0]):
#             logits = out[j, ...]
#             pred, raw_pred = string_utils.naive_decode(logits)
#             pred_str = string_utils.label2str(pred, idx_to_char, False)
#             gt_str = x['gt'][j]
#             cer = error_rates.cer(gt_str, pred_str)
#             wer = error_rates.wer(gt_str, pred_str)
#             gt = gt_str
#             ot = pred_str
#             sum_loss += cer
#             sum_wer_loss += wer
#             steps += 1
#         disp_loss = sum_loss / steps
#     eTime = time.time() - startTime
#     message = message + "\n" + "Epoch: " + str(epoch) + " Training CER: " + str(
#         sum_loss / steps) + " Training WER: " + str(sum_wer_loss / steps) + "\n" + "Time: " + str(
#         eTime) + " Seconds"
#     print("Epoch: " + str(epoch) + " Training CER", sum_loss / steps)
#     print("Training WER: " + str(sum_wer_loss / steps))
#     print("Time: " + str(eTime) + " Seconds")
#     sum_loss = 0.0
#     sum_wer_loss = 0.0
#     steps = 0.0
#     hw.eval()
#     print("Validation Set Size = " + str(len(test_dataloader)))
#     for x in tqdm(test_dataloader):
#         torch.no_grad()
#         line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
#         # labels =  Variable(x['labels'], requires_grad=False, volatile=True)
#         # label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
#         preds = hw(line_imgs).cpu()
#         out = preds.data.cpu().numpy()
#         for i, gt_line in enumerate(x['gt']):
#             logits = out[i, ...]
#             pred, raw_pred = string_utils.naive_decode(logits)
#             pred_str = string_utils.label2str(pred, idx_to_char, False)
#             cer = error_rates.cer(gt_line, pred_str)
#             wer = error_rates.wer(gt_line, pred_str)
#             sum_wer_loss += wer
#             sum_loss += cer
#             steps += 1

#     message = message + "\nTest CER: " + str(sum_loss / steps)
#     message = message + "\nTest WER: " + str(sum_wer_loss / steps)
#     print("Test CER", sum_loss / steps)
#     print("Test WER", sum_wer_loss / steps)
#     best_distance += 1
#     metric = "CER"
#     if (metric == "CER"):
#         if lowest_loss > sum_loss / steps:
#             lowest_loss = sum_loss / steps
#             print("Saving Best")
#             message = message + "\nBest Result :)"
#             torch.save(hw.state_dict(), os.path.join(model_save_path + str(epoch) + ".pt"))
#             best_distance = 0
#         if best_distance > 800:
#             break
#     elif (metric == "WER"):
#         if lowest_loss > sum_wer_loss / steps:
#             lowest_loss = sum_wer_loss / steps
#             print("Saving Best")
#             message = message + "\nBest Result :)"
#             torch.save(hw.state_dict(), os.path.join(model_save_path + str(epoch) + ".pt"))
#             best_distance = 0
#         if best_distance > 80:
#             break
#     else:
#         print("This is actually very bad")ort json
# imp


# In[ ]:


# sam's new main here

import json
# import character_set
import sys
# import hwpsych_dataset
# from hwpsych_dataset import HwDataset
# import model.urnn, urnn2, urnn_window
# import model.crnn_5 as crnn
# import crnn2
import os
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
# import error_rates
# import string_utils
import time
from torch.nn.modules.loss import CTCLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class PsychCTC(torch.nn.Module):
    def __init__(self, idx_to_char, char_to_idx, verbose=False):
        super(PsychCTC, self).__init__()
        self.citerion = CTCLoss(reduction='none', zero_infinity=True)
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.verbose = verbose

    def forward(self, preds, labels, preds_size, label_lengths, psych):
        # print(len(self.char_to_idx))
        loss = self.citerion(preds, labels, preds_size, label_lengths).cuda()
        index = 0
        # lbl = labels.detach().cpu().numpy()
        lbl = labels.data.cpu().numpy()
        lbl_len = label_lengths.data.cpu().numpy()
        output_batch = preds.permute(1, 0, 2)
        # out = output_batch.detach().cpu().numpy()
        out = output_batch.data.cpu().numpy()
        cer = torch.zeros(loss.shape)
        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = naive_decode(logits)
            pred_str = label2str(pred, self.idx_to_char, False)
            gt_str = label2str(lbl[index:lbl_len[j] + index], self.idx_to_char, False)
            index += lbl_len[j]
            # tensor object is not callable?

            #             print(type(gt_str))
            #             print(type(pred_str))
            cer[j] = error_cer(gt_str, pred_str)
            # if self.verbose or psych[j] > 0:
            #     print(psych[j])
        cer = Variable(cer, requires_grad=True).cuda()
        loss = loss + (psych * cer)
        return torch.sum(loss)


# config_path = sys.argv[1]
# try:
#     jobID = sys.argv[2]
# except:
#     jobID = ""
# print(jobID)

# with open(config_path) as f:
#     config = json.load(f)


model_save_path = config['model_save_path']

verbose = False

dirname = os.path.dirname(model_save_path)

# print(dirname)
# if len(dirname) > 0 and not os.path.exists(dirname):
#     os.makedirs(dirname)

# with open(config_path) as f:
#     paramList = f.readlines()

# for x in paramList:
#     print(x[:-1])

baseMessage = "foo"

# for line in paramList:
#     baseMessage = baseMessage + line

# print(baseMessage)

idx_to_char, char_to_idx = load_char_set(config['character_set_path'])

train_dataset = HwDataset(config['training_set_path'],
                          char_to_idx,
                          img_height=config['network']['input_height'],
                          root_path=config['image_root_directory'],
                          augmentation=False,
                          psychPath=psychPath,
                          randomW=False)

try:
    test_dataset = HwDataset(config['validation_set_path'],
                             char_to_idx,
                             img_height=config['network']['input_height'],
                             root_path=config['image_root_directory'],
                             augmentation=False,
                             psychPath=psychPath,
                             randomW=False)

except KeyError as e:
    print("No validation set found, generating one")
    master = train_dataset
    print("Total of " + str(len(master)) + " Training Examples")
    n = len(master)  # how many total elements you have
    n_test = int(n * .1)
    n_train = n - n_test
    idx = list(range(n))  # indices to all elements
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    test_dataset = data.Subset(master, test_idx)
    train_dataset = data.Subset(master, train_idx)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
                              collate_fn=batch_collate)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0,
                             collate_fn=batch_collate)
print("Train Dataset Length: " + str(len(train_dataset)))
print("Test Dataset Length: " + str(len(test_dataset)))

hw = create_model({
    'input_height': config['network']['input_height'],
    'cnn_out_size': config['network']['cnn_out_size'],
    'num_of_channels': 3,
    'num_of_outputs': len(idx_to_char) + 1
})

# if config['model'] == "crnn":
#     print("Using CRNN")
#     hw = crnn.create_model({
#         'input_height': config['network']['input_height'],
#         'cnn_out_size': config['network']['cnn_out_size'],
#         'num_of_channels': 3,
#         'num_of_outputs': len(idx_to_char) + 1
#     })
# elif config['model'] == "urnn":
#     print("Using URNN")
#     hw = urnn.create_model({
#         'input_height': config['network']['input_height'],
#         'cnn_out_size': config['network']['cnn_out_size'],
#         'num_of_channels': 3,
#         'num_of_outputs': len(idx_to_char) + 1,
#         'bridge_width': config['network']['bridge_width']
#     })
# elif config['model'] == "urnn2":
#     print("Using URNN with Curtis's recurrence")
#     hw = urnn2.create_model({
#         'input_height': config['network']['input_height'],
#         'cnn_out_size': config['network']['cnn_out_size'],
#         'num_of_channels': 3,
#         'num_of_outputs': len(idx_to_char) + 1,
#         'bridge_width': config['network']['bridge_width']
#     })
# elif config['model'] == "crnn2":
#     print("Using original CRNN")
#     hw = crnn2.create_model({
#         'cnn_out_size': config['network']['cnn_out_size'],
#         'num_of_channels': 3,
#         'num_of_outputs': len(idx_to_char) + 1
#     })
# elif config['model'] == "urnn3":
#     print("Using windowed URNN with Curtis's recurrence")
#     hw = urnn_window.create_model({
#         'input_height': config['network']['input_height'],
#         'cnn_out_size': config['network']['cnn_out_size'],
#         'num_of_channels': 3,
#         'num_of_outputs': len(idx_to_char) + 1,
#         'bridge_width': config['network']['bridge_width']
#     })

resume = True
try:
    config['model_load_path']
except KeyError:
    resume = False

pretrain = True
try:
    config['pretrained_load_path']
except KeyError:
    pretrain = False

freeze = True
try:
    config['freeze_pretrained_weights']
except KeyError:
    freeze = False

print("Freeze: " + str(freeze))
metric = "CER"
try:
    metric = config['metric'].upper()
except KeyError:
    print("No metric listed, defaulting to Character Error Rate")
print("Metric: " + metric)

if resume:
    hw.load_state_dict(torch.load(config['model_load_path']))
elif pretrain:
    print("loading pretrained weights")
    model_param = hw.state_dict()
    unet_param = torch.load(config['pretrained_load_path'])
    for i in unet_param.keys():
        model_param['UNet.' + i] = unet_param[i]
    hw.load_state_dict(model_param)

if freeze:
    for param in hw.UNet.parameters():
        param.requires_grad = False

if torch.cuda.is_available():
    hw.cuda()
    dtype = torch.cuda.FloatTensor
    print("Using GPU")
else:
    dtype = torch.FloatTensor
    print("No GPU detected")

# optimizer = torch.optim.Adadelta(hw.parameters(), lr=config['network']['learning_rate'])
optimizer = torch.optim.RMSprop(hw.parameters(), lr=config['network']['learning_rate'])
lmbda = lambda epoch: 0.6
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
criterion = PsychCTC(idx_to_char, char_to_idx, verbose=True)
criterion2 = CTCLoss(reduction='sum', zero_infinity=True)
lowest_loss = float('inf')
best_distance = 0
last_drop = 1
for epoch in range(100000):
    torch.enable_grad()
    startTime = time.time()
    message = baseMessage
    sum_loss = 0.0
    sum_wer_loss = 0.0
    steps = 0.0
    hw.train()
    disp_ctc_loss = 0.0
    disp_loss = 0.0
    gt = ""
    ot = ""
    loss = 0.0
    # train_dataloader = torch.utils.data.random_split(train_dataloader, [16, len(train_dataloader) - 16])[0]
    print("Train Set Size = " + str(len(train_dataloader)))
    prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, x in prog_bar:
        # message = str("CER: " + str(disp_loss) +"\nGT: " +gt +"\nex: "+out+"\nProgress")
        prog_bar.set_description(f'CER: {disp_loss} CTC: {loss} Ground Truth: |{gt}| Network Output: |{ot}|')
        line_imgs = x['line_imgs']
        psych = Variable(torch.Tensor(x['psychs']).type(dtype), requires_grad=True)
        rem = line_imgs.shape[3] % 32
        if rem != 0:
            imgshape = line_imgs.shape
            temp = torch.zeros(imgshape[0], imgshape[1], imgshape[2], imgshape[3] + (32 - rem))
            temp[:, :, :, :imgshape[3]] = line_imgs
            line_imgs = temp
            del temp
        line_imgs = Variable(line_imgs.type(dtype), requires_grad=False)

        labels = Variable(x['labels'], requires_grad=False)
        label_lengths = Variable(x['label_lengths'], requires_grad=False)
        optimizer.zero_grad()
        preds = hw(line_imgs).cpu()
        preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))

        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()
        loss = criterion(preds, labels, preds_size, label_lengths, psych)
        # print(loss)
        loss.backward()
        clip_grad_norm_(hw.parameters(), 50)
        optimizer.step()
        # if i == 0:
        #    for i in xrange(out.shape[0]):
        #        pred, pred_raw = string_utils.naive_decode(out[i,...])
        #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
        #        print(pred_str)

        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = naive_decode(logits)
            pred_str = label2str(pred, idx_to_char, False)
            gt_str = x['gt'][j]
            cer = error_cer(gt_str, pred_str)
            wer = error_wer(gt_str, pred_str)
            gt = gt_str
            ot = pred_str
            sum_loss += cer
            sum_wer_loss += wer
            steps += 1
        disp_loss = sum_loss / steps
    eTime = time.time() - startTime
    message = message + "\n" + "Epoch: " + str(epoch) + " Training CER: " + str(
        sum_loss / steps) + " Training WER: " + str(sum_wer_loss / steps) + "\n" + "Time: " + str(
        eTime) + " Seconds"
    print("Epoch: " + str(epoch) + " Training CER", sum_loss / steps)
    print("Training WER: " + str(sum_wer_loss / steps))
    print("Time: " + str(eTime) + " Seconds")
    sum_loss = 0.0
    sum_wer_loss = 0.0
    sum_ctc_loss = 0.0
    steps = 0.0
    hw.eval()
    print("Validation Set Size = " + str(len(test_dataloader)))
    for x in tqdm(test_dataloader):
        torch.no_grad()
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False, volatile=True)
        label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
        psych = Variable(torch.Tensor(x['psychs']).type(dtype), requires_grad=False)
        preds = hw(line_imgs).cpu()
        preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
        print("--------------")
        print(preds.shape)
        print(psych)
        ctc = criterion2(preds, labels, preds_size, label_lengths).cpu().detach()
        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()
        for i, gt_line in enumerate(x['gt']):
            logits = out[i, ...]
            pred, raw_pred = naive_decode(logits)
            pred_str = label2str(pred, idx_to_char, False)
            cer = error_cer(gt_line, pred_str)
            wer = error_wer(gt_line, pred_str)
            sum_wer_loss += wer
            sum_loss += cer
            sum_ctc_loss += ctc
            steps += 1
    message = message + "\nTest CTC: " + str(sum_ctc_loss.numpy() / steps)
    message = message + "\nTest CER: " + str(sum_loss / steps)
    message = message + "\nTest WER: " + str(sum_wer_loss / steps)
    print("Test CER", sum_loss / steps)
    print("Test WER", sum_wer_loss / steps)
    print("Test CTC", sum_ctc_loss.numpy() / steps)
    best_distance += 1
    metric = "CER"
    if (metric == "CER"):
        if lowest_loss > sum_loss / steps:
            lowest_loss = sum_loss / steps
            print("Saving Best")
            message = message + "\nBest Result :)"
            torch.save(hw.state_dict(), os.path.join(model_save_path + str(epoch) + ".pt"))
            #             email_update(message, jobID)
            best_distance = 0
            last_drop = 1
        if best_distance / 15 > last_drop:
            last_drop += 1
            scheduler.step()
        if best_distance > 80:
            break
    elif (metric == "WER"):
        if lowest_loss > sum_wer_loss / steps:
            lowest_loss = sum_wer_loss / steps
            print("Saving Best")
            message = message + "\nBest Result :)"
            torch.save(hw.state_dict(), os.path.join(model_save_path + str(epoch) + ".pt"))
            #             email_update(message, jobID)
            best_distance = 0
            last_drop = 1
        if best_distance / 15 > last_drop:
            last_drop += 1
            scheduler.step()
        if best_distance > 80:
            break
    elif (metric == "CTC"):
        if lowest_loss > sum_ctc_loss / steps:
            lowest_loss = sum_ctc_loss / steps
            print("Saving Best")
            message = message + "\nBest Result :)"
            torch.save(hw.state_dict(), os.path.join(model_save_path + str(epoch) + ".pt"))
            #             email_update(message, jobID)
            best_distance = 0
            last_drop = 1
        if best_distance / 15 > last_drop:
            last_drop += 1
            scheduler.step()
        if best_distance > 80:
            break
    else:
        print("This is actually very bad")

