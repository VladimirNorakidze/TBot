import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
import time

from train_test_func import *
#from DRAFT import *


# n - номер последнего слоя
def change_classifier(net, data_list, n, last_layer_ftrs):
    mods = nn.Sequential()
    layer_idx = -1
    for m in net.classifier.modules():
        if layer_idx==-1:
            layer_idx += 1
            continue
        if layer_idx==n:
            mods.add_module(str(layer_idx), nn.Linear(last_layer_ftrs, data_list[3]))
            break
        mods.add_module(str(layer_idx), copy.deepcopy(m))
        layer_idx += 1
    print(mods)
    net.classifier = mods
    
    return net


# класс для срезания слоев для структуры сети: (features):{...}, (classifier):{...}
class FeaturesExtractor(nn.Module):
    
    def __init__(self, input_net, layer_idx, classifier=False):
            
        super(FeaturesExtractor, self).__init__()
        mods = OrderedDict()

        feat_len = len(input_net.features)
        if classifier:
            for i in range(0, feat_len):
                mods[str(i)] = copy.deepcopy(input_net.features[i])
                
            for i in range(feat_len, feat_len+layer_idx+1):
                mods[str(i)] = copy.deepcopy(input_net.classifier[feat_len-i])
            
        else:
            for i in range(0, layer_idx):
                mods[str(i)] = copy.deepcopy(input_net.features[i])
                
                
        self.features = nn.Sequential(mods)           
            
    def forward(self, input):
        return self.features(input)

# класс для срезания слоев для структуры сети: (куча слоев), (fc)
class ResnetFeaturesExtractor(nn.Module):
    
    def __init__(self, input_net, classifier=False):
            
        super(ResnetFeaturesExtractor, self).__init__()
        mods = OrderedDict()

        mods['conv1'] = copy.deepcopy(input_net.conv1)
        mods['bn1'] = copy.deepcopy(input_net.bn1)
        mods['relu'] = copy.deepcopy(input_net.relu)
        mods['maxpool'] = copy.deepcopy(input_net.maxpool)
        mods['layer1'] = copy.deepcopy(input_net.layer1)
        mods['layer2'] = copy.deepcopy(input_net.layer2)
        mods['layer3'] = copy.deepcopy(input_net.layer3)
        mods['layer4'] = copy.deepcopy(input_net.layer4)
        mods['avgpool'] = copy.deepcopy(input_net.avgpool)
              
                
        self.features = nn.Sequential(mods)      

	
        if classifier:
            self.fc = copy.deepcopy(input_net.fc)     
            
    def forward(self, input):
        return self.features(input)  		
		
# ALEXNET

# загрузка сети с обученными весами; i-номер датасета в data_test
def load_alexnet(data_list, classifier = True, csv_file=True):
    net = models.alexnet(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False  

    # добавление посленего слоя
    n = len(net.classifier) - 1   # номер последнего слоя 
    last_layer_ftrs = net.classifier[(n)].in_features
    
    net = change_classifier(net, data_list=data_list, n=n, last_layer_ftrs=last_layer_ftrs)
    #num_of_layer = '{}'.format(n+m1)
    #net.classifier.add_module(num_of_layer, nn.Linear(last_layer_ftrs, data_list[3]))   #добавление нового слоя
    
    # загрузка весов
    if csv_file:
        filename = 'alexnet_'+data_list[0]
        net.load_state_dict(torch.load(file_path+filename))
    
    if classifier:
        return net
    else:
        # снимаем последние слои
        buff_net = FeaturesExtractor(net, len(net.features))    
        return buff_net
    
    
# need_train - нужно ли дообучать сеть; num_datasets - число наборов
# возвращает сеть net - обученную или нет
def alexnet_run(num_datasets, need_train = False, net = None, num_epochs=[25]):
    l = [x for x in range(num_datasets)]
    df = None
    # range(res_losses_before.shape[0])
    for i in l:
        start = time.time()
        
        if (net is None)or(i>0):
            # загрузка сети
            net = models.alexnet(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False  

            # добавление посленего слоя
            n = len(net.classifier) - 1   # номер последнего слоя 
            last_layer_ftrs = net.classifier[(n)].in_features
            
            net = change_classifier(net, data_list=data_test[i], n=n, last_layer_ftrs=last_layer_ftrs)
            #num_of_layer = '{}'.format(n+m1)
            #net.classifier.add_module(num_of_layer, nn.Linear(last_layer_ftrs, data_test[i][3]))   #добавление нового слоя
            print(data_test[i][0], "\n", net.classifier)

        if use_gpu:
            net = net.cuda(CUDA_DEVICE)    
        
        crit = nn.CrossEntropyLoss()
        if use_gpu:
            crit = crit.cuda(CUDA_DEVICE)

        loss, acc = 0.0, 0.0
        
        if need_train:
            model_name = 'alexnet_'+data_train[i][0]
            # Observe that only parameters of final layer are being optimized as opoosed to before.
            optimizer_conv = optim.SGD(net.classifier[(6)].parameters(), lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.m1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            
            if (type(data_train[i][2]) is dict):
                net, loss, acc, df = train_model_div(net, model_name, data_train[i][2], data_train[i][4], crit,
                                                 optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            else:
                net, loss, acc, df = train_model_not_div(net, model_name, data_train[i][2], crit,
                                                     optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            print("TRAIN:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
            
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            print("TEST:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
        else:
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            # data_test[i].append(y_hat)
            
            timer = time.time()-start
            print("JUST TEST:\nBEST LOSS: {}  BEST ACC: {}\nTIME: {}".format(loss, acc, timer))       
            
        

        #bot = telepot.Bot('567990957:AAE40n4HsPyvvWGKtsUAZ6AFqyWL8ksKMyQ')
        #bot.sendMessage(393484655, '{}\n Finish!\nLoss: {:.5}; acc: {:.5}\nTime: {:.3} min'.format(data_test[i][0], loss, acc, timer/60))
        
    return net, df
	
	

	
# VGG

def load_vgg(data_list, num_net, classifier = True, csv_file=True):
    net = None
    if (num_net == 16):
        net = models.vgg16(pretrained=True)
    elif(num_net == 19):
        net = models.vgg19(pretrained=True)
    else:
        print('ERROR: wrong number of vgg!!!')
        return net
    for param in net.parameters():
        param.requires_grad = False  

    # добавление посленего слоя
    n = len(net.classifier) - 1   # номер последнего слоя 
    last_layer_ftrs = net.classifier[(n)].in_features
    
    net = change_classifier(net, data_list=data_list, n=n, last_layer_ftrs=last_layer_ftrs)
    
    # загрузка весов
    if csv_file:
        filename = 'vgg{}_{}'.format(num_net, data_list[0])
        net.load_state_dict(torch.load(file_path+filename))
    
    if classifier:
        return net
    else:
        # снимаем последние слои
        buff_net = FeaturesExtractor(net, len(net.features))    
        return buff_net

      
def vgg_run(num_datasets, num_net, need_train = False, net = None, num_epochs=25):
    l = [x for x in range(num_datasets)]
    #l = [num_datasets]
    df = None
    # range(res_losses_before.shape[0])
    for i in l:
        start = time.time()
        
        if (net is None)or(i>0):
            # загрузка сети
            if (num_net == 16):
                net = models.vgg16(pretrained=True)
            elif (num_net == 19):
                net = models.vgg19(pretrained=True)
            else:
                print('ERROR: wrong net number!!!')
                return net, df
            
            for param in net.parameters():
                param.requires_grad = False  

            # добавление посленего слоя
            n = len(net.classifier) - 1   # номер последнего слоя 
            last_layer_ftrs = net.classifier[(n)].in_features
            net = change_classifier(net, data_list=data_test[i], n=n, last_layer_ftrs=last_layer_ftrs)
            print(data_train[i][0], "\n", net.classifier[(6)])

        if use_gpu:
            net = net.cuda(CUDA_DEVICE)    
        
        crit = nn.CrossEntropyLoss()
        if use_gpu:
            crit = crit.cuda(CUDA_DEVICE)

        loss, acc = 0.0, 0.0
        
        if need_train:
            model_name = 'vgg{}_{}'.format(num_net, data_train[i][0])
            print(model_name)
            # Observe that only parameters of final layer are being optimized as opoosed to before.
            optimizer_conv = optim.SGD(net.classifier[(6)].parameters(), lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.m1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            
            if (type(data_train[i][2]) is dict):
                net, loss, acc, df = train_model_div(net, model_name, data_train[i][2], data_train[i][4], crit,
                                                 optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            else:
                net, loss, acc, df = train_model_not_div(net, model_name, data_train[i][2], crit,
                                                     optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            print("TRAIN:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
            
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            print("TEST:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
        else:
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            # data_test[i].append(y_hat)
            
            timer = time.time()-start
            print("JUST TEST:\nBEST LOSS: {}  BEST ACC: {}\nTIME: {}".format(loss, acc, timer))       
            
        

        #bot = telepot.Bot('567990957:AAE40n4HsPyvvWGKtsUAZ6AFqyWL8ksKMyQ')
        #bot.sendMessage(393484655, '{}\n Finish!\nLoss: {:.5}; acc: {:.5}\nTime: {:.3} min'.format(data_test[i][0], loss, acc, timer/60))
        
    return net, df
	

	

# RESNET
def load_resnet(data_list, num_net, classifier = True, csv_file=True, num_labels=1000):
    net = None
    if (num_net == 50):
        net = models.resnet50(pretrained=True)
    elif(num_net == 101):
        net = models.resnet101(pretrained=True)
    elif(num_net == 34):
        net = models.resnet34(pretrained=True)    
    else:
        print('ERROR: wrong number of resnet!!!')
        return net
    for param in net.parameters():
        param.requires_grad = False  

    # замена посленего слоя
    last_layer_ftrs = net.fc.in_features
    net.fc = nn.Linear(last_layer_ftrs, num_labels)
    
    # загрузка весов
    if csv_file:
        filename = 'resnet{}_{}'.format(num_net, data_list[0])
        print(file_path+filename)
        net.load_state_dict(torch.load(file_path+filename))
    
    if classifier:
        return net
    else:
        # снимаем последние слои
        buff_net = ResnetFeaturesExtractor(net)    
        return buff_net 
    
    
def resnet_run(num_datasets, num_resnet, data_test, need_train = False, net = None, num_epochs=25):
    l = [x for x in range(num_datasets)]
    df = None
    # range(res_losses_before.shape[0])
    for i in l:
        start = time.time()
        
        if (net is None):
            # загрузка сети
            if (num_resnet == 50):
                net = models.resnet50(pretrained=True)
            if (num_resnet == 101):
                net = models.resnet101(pretrained=True)
            if (num_resnet == 34):
                net = models.resnet34(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False  
  
            # добавление посленего слоя
            last_layer_ftrs = net.fc.in_features
            net.fc = nn.Linear(last_layer_ftrs, data_test[i][3])
            #net.add_module('fc2', nn.Linear(last_layer_ftrs, data_test[i][3]))   #добавление нового слоя
            print(net.fc)

        if use_gpu:
            net = net.cuda(CUDA_DEVICE)    
        
        crit = nn.CrossEntropyLoss()
        if use_gpu:
            crit = crit.cuda(CUDA_DEVICE)

        loss, acc = 0.0, 0.0
        
        if need_train:
            model_name = 'resnet{}_{}'.format(num_resnet, data_train[i][0])
            print(model_name)
            # Observe that only parameters of final layer are being optimized as opoosed to before.
            optimizer_conv = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.m1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            
            if (type(data_train[i][2]) is dict):
                net, loss, acc, df = train_model_div(net, model_name, data_train[i][2], data_train[i][4], crit,
                                                 optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            else:
                net, loss, acc, df = train_model_not_div(net, model_name, data_train[i][2], crit,
                                                     optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            print("TRAIN:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
            
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            print("TEST:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
        else:
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            # data_test[i].append(y_hat)
            
            timer = time.time()-start
            print("JUST TEST:\nBEST LOSS: {}  BEST ACC: {}\nTIME: {}".format(loss, acc, timer))       
            
        

        #bot = telepot.Bot('567990957:AAE40n4HsPyvvWGKtsUAZ6AFqyWL8ksKMyQ')
        #bot.sendMessage(393484655, '{}\n Finish!\nLoss: {:.5}; acc: {:.5}\nTime: {:.3} min'.format(data_test[i][0], loss, acc, timer/60))
        
    return net, df
	
	

	
# DENSENET
def load_densenet(data_list, num_net, classifier = True, csv_file=True):
    net = None
    if (num_net == 121):
        net = models.densenet121(pretrained=True)
    elif(num_net == 161):
        net = models.densenet161(pretrained=True)
    elif(num_net == 169):
        net = models.densenet169(pretrained=True)
    else:
        print('ERROR: wrong number of densenet!!!')
        return net
    for param in net.parameters():
        param.requires_grad = False  

    # добавление посленего слоя
    last_layer_ftrs = net.classifier.in_features
    net.classifier = nn.Linear(last_layer_ftrs, data_list[3])
    
    # загрузка весов
    if csv_file:
        filename = 'densenet{}_{}'.format(num_net, data_list[0])
        print(file_path+filename)
        net.load_state_dict(torch.load(file_path+filename))
    
    if classifier:
        return net
    else:
        # снимаем последние слои
        buff_net = FeaturesExtractor(net, len(net.features))    
        return buff_net    
    
    
def dense_run(num_datasets, num_net, need_train = False, net = None, num_epochs=25):
    l = [x for x in range(num_datasets)]
    df = None
    # range(res_losses_before.shape[0])
    for i in l:
        print(data_test[i][0])
        start = time.time()
        
        if (net is None)or(i>0):
            # загрузка сети
            if (num_net == 121):
                net = models.densenet121(pretrained=True)
            elif (num_net == 161):
                net = models.densenet161(pretrained=True)
            elif (num_net == 169):
                net = models.densenet169(pretrained=True)    
            else:
                print('ERROR: wrong net number!!!')
                return net, df
            
            for param in net.parameters():
                param.requires_grad = False  
  
            # добавление посленего слоя
            last_layer_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(last_layer_ftrs, data_test[i][3])
            #net.add_module('fc2', nn.Linear(last_layer_ftrs, data_test[i][3]))   #добавление нового слоя
            print(net.classifier)

        if use_gpu:
            net = net.cuda(CUDA_DEVICE)    
        
        crit = nn.CrossEntropyLoss()
        if use_gpu:
            crit = crit.cuda(CUDA_DEVICE)

        loss, acc = 0.0, 0.0
        
        if need_train:
            model_name = 'densenet{}_{}'.format(num_net, data_train[i][0])
            print(model_name)
            # Observe that only parameters of final layer are being optimized as opoosed to before.
            optimizer_conv = optim.SGD(net.classifier.parameters(), lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.m1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            
            if (type(data_train[i][2]) is dict):
                net, loss, acc, df = train_model_div(net, model_name, data_train[i][2], data_train[i][4], crit,
                                                 optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            else:
                net, loss, acc, df = train_model_not_div(net, model_name, data_train[i][2], crit,
                                                     optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs[i])
            print("TRAIN:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
            
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            print("TEST:\nBEST LOSS: {}  BEST ACC: {}\n".format(loss, acc))
        else:
            loss, acc, y_hat = test_model(net, data_test[i][2], crit)
            # data_test[i].append(y_hat)
            
            timer = time.time()-start
            print("JUST TEST:\nBEST LOSS: {}  BEST ACC: {}\nTIME: {}".format(loss, acc, timer))      
            
        

        #bot = telepot.Bot('567990957:AAE40n4HsPyvvWGKtsUAZ6AFqyWL8ksKMyQ')
        #bot.sendMessage(393484655, '{}\n Finish!\nLoss: {:.5}; acc: {:.5}\nTime: {:.3} min'.format(data_test[i][0], loss, acc, timer/60))
        
    return net, df
	
	
	
def load_net(data_list, name_net, num_net, classifier = True, csv_file=True):
	buf = None
	if (name_net=='alexnet'):
		buf = load_alexnet(data_list=data_list, classifier = classifier, csv_file=csv_file)
	elif (name_net=='vgg'):
		buf = load_vgg(data_list=data_list, num_net=num_net, classifier = classifier, csv_file=csv_file)
	elif (name_net=='resnet'):
		buf = load_resnet(data_list=data_list, num_net=num_net, classifier = classifier, csv_file=csv_file)
	elif (name_net=='densenet'):
		buf = load_densenet(data_list, num_net=num_net, classifier = classifier, csv_file=csv_file)
	else:
		print('WRONG NET NAME')
		return 0
        
	return buf


