import torch
import numpy as np

def labelToOneHot(ids,classnum = 7):
    '''
    将标签列表转化为one-hot 列表
    :param ids: label list or array
    :return:
    '''
    if not isinstance(ids,(list,np.ndarray)):
        raise ValueError('ids should be 1-d list or array')
    a = torch.LongTensor(len(ids),classnum).zero_()
    ids = torch.LongTensor(ids).view(-1,1)
    b = a.scatter_(1,ids,value=1.)
    return b