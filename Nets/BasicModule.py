import torch as t
import time

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module，主要是提供save和load两个方法
    '''
    def __init__(self):
        super(BasicModule,self).__init__()

    def load(self,path):
        '''
        加载保存好的模型
        :param path: 模型保存路径
        :return:
        '''
        self.load_state_dict(t.load(path))

    def save(self,name = None):
        '''
        保存模型，使用时间命名模型
        :param name:
        :return:
        '''
        if name is None:
            prefix = 'checkpoint/'
            name = time.strftime(prefix + '%m%d_%H%M%S.pth')
        t.save(self.state_dict(),name)
        return name

class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)