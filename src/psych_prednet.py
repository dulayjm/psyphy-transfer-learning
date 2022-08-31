
# Justin Dulay

# prednet implementation for imagenet + rt data in pytorch/pytorch lightning



from argparse import ArgumentParser
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback, seed_everything
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import json

from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn

from PIL import Image

from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor,
                                    Lambda
                                   )


# reaction time psychophysical loss
def RtPsychCrossEntropyLoss(outputs, targets, psych):
#     print('in psych loss')
#     print(type(targets))
#     print(type(outputs))
#     print('the outputs are', outputs)

    targets = targets.cpu().detach().numpy()

    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    
#     print('in loss', targets)
#     new_fucks = []
#     for elem in targets: 
# #         print('the elem is .... ', elem)
#         elem = label2id[elem]
# #         print('the elem is now .... ', elem)
#         new_fucks.append(int(elem))
    
#     targets = np.asarray([id2label[i] for i in targets])
#     targets = torch.as_tensor(targets)
#     print('here', new_fucks)
#     targets = np.asarray(new_fucks)
    
    # converting reaction time to penalty
    # 10002 is close to the max penalty time seen in the data
    for idx in range(len(psych)):   
        psych[idx] = abs(28 - psych[idx]) 
        # seems to be in terms of 10 for now,
        # will fix later

    # adding penalty to each of the output logits 
    for i in range(len(outputs)):
#         print('psych[i]', psych[i])
        val = psych[i] / 30
            
        outputs[i] += val 

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs) / num_examples

def _softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def _log_softmax(x):
    return torch.log(_softmax(x))


# In[20]:


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    rt = torch.tensor([x["rt"] for x in batch])

    return {"pixel_values": pixel_values, "label": labels, "rt": rt}


# In[21]:




# In[22]:


class ReactionTimeDataset(Dataset):
    def __init__(self,
                 json_path,
                 transform):

        with open(json_path) as f:
            data = json.load(f)
        #print("Json file loaded: %s" % json_path)

        self.data = data
        self.transform = transform
        self.random_weight = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[str(idx)]

        # Open the image and do normalization and augmentation
        img = Image.open(item["img_path"])
        img = img.convert('RGB')
        # needed this transform call
        img = self.transform(img)
        
        # Deal with reaction times
        if item["RT"] != None:
            rt = item["RT"]
        else:
            rt = 0

        return {
            "pixel_values": img,
            "label": item["label"],
            "rt": rt,
            "category": item["category"]
        }


# In[23]:


import torch
import math
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

# https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = Parameter(torch.Tensor(
            4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(
            4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = Parameter(torch.Tensor(
            3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx
        wx = F.conv2d(input, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                      self.padding_h, self.dilation, self.groups)

        # Cell uses a Hadamard product instead of a convolution?
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                      self.padding_h, self.dilation, self.groups)

        wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(
            wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)

        i = F.sigmoid(wxhc[:, :self.out_channels])
        f = F.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = F.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
        o = F.sigmoid(wxhc[:, 3 * self.out_channels:])

        c_1 = f * c_0 + i * g
        h_1 = o * F.tanh(c_1)
        return h_1, (h_1, c_1)


# In[24]:


def info(prefix, var):
    print('-------{}----------'.format(prefix))
    if isinstance(var, torch.autograd.variable.Variable):
        print('Variable:')
        print('size: ', var.data.size())
        print('data type: ', type(var.data))
    elif isinstance(var, torch.FloatTensor) or isinstance(var, torch.cuda.FloatTensor):
        print('Tensor:')
        print('size: ', var.size())
        print('type: ', type(var))
    else:
        print(type(var))


# In[25]:


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable



class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error'):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.output_mode = output_mode

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1],                                                                             self.r_channels[i],
                                (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input, rts=None):

        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h)).cuda()
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)).cuda()
            w = w//2
            h = h//2
        time_steps = input.size(1)
        total_error = []
        
        for t in range(time_steps):
            A = input[:,t]
            A = A.type(torch.cuda.FloatTensor)
            
            # dummy scalar loss for now
            
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R, hx = cell(tmp, hx)
                R_seq[l] = R
                H_seq[l] = hx


            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if torch.mean(rts) > 8:
                    A_hat = torch.mul(A_hat, 0.95)
                if l == 0:
                    frame_prediction = A_hat
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2) # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return frame_prediction


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'min_val=' + str(self.lower) + ', max_val=' + str(self.upper) + inplace_str + ')'


# In[26]:


import torchmetrics

class PredNetLightningModule(pl.LightningModule):
    def __init__(self, backbone, traindataset, valdataset, testdataset):
        super(PredNetLightningModule, self).__init__()
        self.backbone = backbone
        # self.vit = torchvision.models.vit_b_32(pretrained=True) 
#         self.num_labels=336
        self.criterion = nn.CrossEntropyLoss()
#         self.fc = nn.Linear(4096, 336)

#         self.classifier = nn.Linear(self.vit.config.hidden_size, 164)
        self.accuracy = torchmetrics.Accuracy()
        
        self.train_dataset = traindataset
        self.val_dataset = valdataset
        self.test_dataset = testdataset

#     def forward(self, pixel_values):
# #         outputs = self.fc(outputs)
#         outputs = self.vit(pixel_values=pixel_values, return_dict=False)
#         print('type of outputs ', outputs[0].shape)
# #         logits = self.classifier(outputs[0])

#         return outputs[0]
#         self.layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]])).to(self.device)
#         self.time_loss_weights = torch.ones(nt, 1)
#         self.time_loss_weights[0] = 0
#         self.time_loss_weights = Variable(self.time_loss_weights).to(self.device)

    def forward(self, x, rts):
        return self.backbone(x, rts)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, num_workers=8, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, num_workers=8, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, num_workers=8, collate_fn=collate_fn)

    
    def common_step(self, batch, batch_idx):        
        inputs = batch['pixel_values']
        labels = batch['label']
        labels = labels.cpu().detach().numpy()

#         temp = []
#         for elem in labels: 
#             elem = label2id[elem]
#             temp.append(int(elem))
    
#         labels = np.asarray(temp)
#         labels = torch.from_numpy(labels).to(self.device)

        rts = batch['rt']
        rts = rts.type(torch.float)

#         print("INFO. feats: {} - labels {} - rts {} --".format(pixel_values, labels, rts))
#         print("INFO. feats: {} - labels {} - rts {} --".format(type(pixel_values), type(labels), type(rts)))
    
    
        # add new dim 
        # orig shape batch x chan x w x h

        inputs = inputs[:,None,:,:,:]
        
#         print('original inputs shape', inputs.shape)
        # batch x time_steps x channel x width x height
        
        
#         inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
        inputs = Variable(inputs.to(self.device))
    
    
        errors = self(inputs, rts)
        loc_batch = errors.size(0)
        
#         print('errors are', errors)

        
#         print('loc_batch', loc_batch)

        # mm = matrix multiplication
        # view = -1 is a *dimension. so it's reshape, but changes the tensor
                
        # so op here is 
#         print('errors shape', errors.shape)
#         print('HERE')
#         print('the op looks like:')
        
#         print('sanity nt is', nt)
        

        errors = errors.to(self.device)
#         self.time_loss_weights = self.time_loss_weights.to(self.device)
#         self.layer_loss_weights = self.layer_loss_weights.to(self.device)
        
        # migtht just need error there
        
#         print(errors.shape, time_loss_weights.shape) 
#         print('errors view is ', errors.view(-1, 1))
        
#         errors = torch.mm(errors.view(-1, 1), self.time_loss_weights) # batch*n_layers x 1
        
        
#         print(errors.shape, layer_loss_weights.shape) 
#         print('errors view is ', errors.view(loc_batch, -1))
        
#         errors = torch.mm(errors.view(loc_batch, -1), self.layer_loss_weights)
        mean_error = torch.mean(errors)
        
#         print('final errors are w/o layer loss or mean', errors)
#         print('final errors shape', errors.shape)
        
        
        
#         1/0
        
#         print("INFO. logits: {} - labels {} - rts {} --".format(logits, labels, rts))
#         print("INFO. ---shapes--- logits: {} - labels {} - rts {} --".format(logits.shape, labels.shape, rts.shape))

#         loss = RtPsychCrossEntropyLoss(logits, labels, rts)
#         loss = self.criterion(logits, labels)
        
#         labels_hat = torch.argmax(logits, dim=1)
#         accuracy = self.accuracy(labels_hat, labels)

        return mean_error
      
    def training_step(self, batch, batch_idx):
        mean_error = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", mean_error)
#         self.log("training_accuracy", accuracy)

        return mean_error
    
    def validation_step(self, batch, batch_idx):
        mean_error = self.common_step(batch, batch_idx)     
        self.log("validation_loss", mean_error, on_epoch=True)
#         self.log("validation_accuracy", accuracy, on_epoch=True)

        return mean_error

    def test_step(self, batch, batch_idx):
        # print('batch is in testing', batch)
        # 1/0
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return torch.optim.Adam(self.parameters(), lr=5e-5)    


# In[27]:


# main ...



# In[28]:
if __name__=='__main__':

    # imagenetmemas
    # normalize = imagenetMeans
    train_transforms = Compose(
            [
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
    #             normalize,
            ]
        )

    #TODO: just split up the dataset fairly 
    # maybe just train/test only
    json_data_base = '/afs/crc.nd.edu/user/j/jdulay'
    train_known_known_with_rt_path = os.path.join(json_data_base, "train_known_known_with_rt.json")
    valid_known_known_with_rt_path = os.path.join(json_data_base, "valid_known_known_with_rt.json")

    traindataset = ReactionTimeDataset(json_path=train_known_known_with_rt_path,
                                            transform=train_transforms)


    valdataset = traindataset
    testdataset = ReactionTimeDataset(json_path=valid_known_known_with_rt_path,
                                            transform=train_transforms)

    testdataset


    # In[29]:


    labels = []
    #TODO might need to do this for train idk
    for i in range(len(testdataset)):
        item = testdataset.data[str(i)]['label']
        if item not in labels:
            labels.append(item)
    len(labels)
    # so we use this as the classes for now


    # In[30]:


    # backbone is ...

    # these may by kitti specific ...
    A_channels = (3, 48, 96, 192)
    R_channels = (3, 48, 96, 192)

    backbone = PredNet(R_channels, A_channels, output_mode='error')


    # In[31]:


    import os
    import wandb

    path = '/afa/crc.nd.edu/user/j/jdulay/.cache/'
    if os.path.isdir(path):
        os.rmdir(path)
        
    # wandb_logger = None
    logger_name = '01_prednet_psych_Ahat'
    wandb_logger = WandbLogger(name=logger_name, project="prednet")

    model = PredNetLightningModule(backbone=backbone,traindataset=traindataset, valdataset=valdataset, testdataset=testdataset)

    print('data len', len(testdataset))

    trainer = pl.Trainer(
        max_epochs=20, 
        devices=1, 
    #     accelerator='gpu',
        gpus=1,
    #     strategy='ddp',
    #     auto_select_gpus=True, 
        logger=wandb_logger,
    #     callbacks=[metrics_callback],
        num_sanity_val_steps=2,
    #     progress_bar_refresh_rate=1000,
    #     limit_train_batches=0,
    #     limit_val_batches=0
    )


    # In[32]:


    trainer.fit(model)

    trainer.save_checkpoint('psych_ckpt_ahat_02.pt')

    # In[33]:


