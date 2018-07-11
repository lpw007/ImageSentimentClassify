from config import opt
import os
import torch
import torch.nn as nn
import torch.optim as Optim
import Nets
from data.dataset import imageSentiment
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import torch.nn.functional as F

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    #step1: config model
    model = getattr(Nets,opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    #step2: data
    train_data = imageSentiment(opt.train_path,train = True) #训练集
    val_data = imageSentiment(opt.train_path,train = False) #验证集
    train_dataloader = DataLoader(train_data,batch_size = opt.batch_size,shuffle=True,num_workers = opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size = opt.batch_size,shuffle=False,num_workers = opt.num_workers)

    #step3: 定义损失函数及优化器
    # criterion = nn.CrossEntropyLoss() #交叉熵损失函数 如果使用该损失函数 则网络最后无需使用softmax函数
    lr = opt.lr
    # optimizer = Optim.Adam(model.parameters(),lr = lr,weight_decay= opt.weight_decay)
    optimizer = Optim.SGD(model.parameters(),lr = 0.001,momentum=0.9,nesterov=True)
    #step4: 统计指标(计算平均损失以及混淆矩阵)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(7)
    previous_loss = 1e100

    #训练
    for i in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        total_loss = 0.
        for ii,(label,data) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            if opt.use_gpu:
                label,data = label.to(device),data.to(device)

            optimizer.zero_grad()
            score = model(data)
            # ps:使用nll_loss和crossentropyloss进行多分类时 target为索引标签即可 无需转为one-hot
            loss = F.nll_loss(score,label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            #更新统计指标以及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data,label.data)

            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss',loss_meter.value()[0])

        vis.plot('mach avgloss', total_loss/len(train_dataloader))
        model.save()

        #计算验证集上的指标
        val_accuracy = val(model,val_dataloader)

        vis.plot('val_accuracy',val_accuracy)

        # #随着误差的降低减慢学习率
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * opt.lr_delay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #
        # previous_loss = loss_meter.value()[0]

def model_test(**kwargs):
    '''
    根据传进来的参数设置加载的模型，将模型的测试加过写入到保存文件中
    '''
    opt.parse(kwargs)
    model = getattr(Nets, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    #使用cuda
    if opt.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    #准备测试数据
    test_data = imageSentiment(opt.train_path, test=True,train=False)  # 训练集
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []

    with torch.no_grad():
        for ii, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            label,input = data
            if opt.use_gpu:
                model.to(device)
                label,input = label.to(device),input.to(device)
            score = model(input)
            _, predicted = torch.max(score.data, 1)
            batch_result = [(int(path_),int(label_)) for path_,label_ in zip(label,score)]
            results += batch_result
    write_csv(results,opt.result_file) #将结果写进CSV文件

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def val(model,dataloader):
    '''
    计算模型在验证集上的正确率
    :param model:
    :param dataloader:
    :return:
    '''
    model.eval() #将模型设置为验证模式

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total = 0
    correct = 0
    with torch.no_grad():
        for ii, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            label,input = data
            if opt.use_gpu:
                model.to(device)
                label,input = label.to(device),input.to(device)
            score = model(input)
            _,predicted = torch.max(score.data,1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
    model.train()
    return float(correct/total)

def help():
    '''
    打印帮助的信息： python file.py help
    '''

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
