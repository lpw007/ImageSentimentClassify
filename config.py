import warnings

class DefaultConfig(object):
    env = 'default'
    model = 'CNN'

    train_path = './data/train.csv'
    test_path = './data/test.csv'
    load_model_path = None

    batch_size = 16
    use_gpu = True
    num_workers = 4
    print_freq = 30

    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_delay = 0.95
    weight_decay = 1e-4

def parse(self,kwargs):
    '''
    根据字典跟新参数
    :param kwargs:
    :return:
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn('warning: opt has no attribute %s' %k)
        setattr(self,k,v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('_'):
            print(k,getattr(self,k))

DefaultConfig.parse = parse
opt = DefaultConfig()