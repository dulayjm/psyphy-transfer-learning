#!/bin/bash

# foo test
# test 2

# test again

import numpy as np
import torch

# reaction time psychophysical loss
def RtPsychCrossEntropyLoss(outputs, targets, psych):
    print('in psych loss')
    print(type(targets))
    print(type(outputs))
    num_examples = targets.shape[0]
    print('the outputs are', outputs)
    batch_size = outputs.size[0]

    # converting reaction time to penalty
    # 10002 is close to the max penalty time seen in the data
    for idx in range(len(psych)):
        psych[idx] = abs(28 - psych[idx])
        # seems to be in terms of 10 for now,
        # will fix later

    # adding penalty to each of the output logits
    for i in range(len(outputs)):
        val = psych[i] / 30
        if np.isnan(val):
            val = 0

        outputs[i] += val

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs) / num_examples

# mean accuracy psychophysical loss
def AccPsychCrossEntropyLoss(outputs, targets, psych):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]

    # converting accuracy to penalty
    for idx in range(len(psych)):
        psych[idx] = abs(1 - psych[idx])

    for i in range(len(outputs)):
        outputs[i] += (psych[i])

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs)/num_examples

def _softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def _log_softmax(x):
    return torch.log(_softmax(x))