import torch
from torch.autograd import Variable

import copy, time
import numpy as np
import pandas as pd




# взято с http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# обучение модели; dataloader разделен на обуч. и валид. выборки
def train_model_div(model, model_name, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    losses, accs = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)          
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
                
            
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda(CUDA_DEVICE))
                    labels = Variable(labels.cuda(CUDA_DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            losses.append(epoch_loss)
            accs.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, file_path+model_name)

        print()

    time_elapsed = time.time() - since
    df = pd.DataFrame({'losses': losses, 'accs': accs})
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc, df
	
	
	
# обучение модели; dataloader один            
def train_model_not_div(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    losses, accs = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)          
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            dataset_sizes = 0
            count = 0
            
            # Iterate over data.
            for data in dataloaders:
                # get the inputs
                inputs, labels = None, None
                if (phase == 'train'):
                    if (count > int(0.3*len(dataloaders.dataset))):
                        inputs, labels = data
                    count += data[0].shape[0]
                if (phase == 'val'):
                    if (count <= int(0.3*len(dataloaders.dataset))):
                        inputs, labels = data
                    count += data[0].shape[0]

                if inputs is None:
                    continue
                    
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda(CUDA_DEVICE))
                    labels = Variable(labels.cuda(CUDA_DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                #print(inputs)
                outputs = model(inputs)
                #print(outputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                dataset_sizes += data_batch_size
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            print(dataset_sizes, data_batch_size, end=' ')
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects / dataset_sizes
            losses.append(epoch_loss)
            accs.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, file_path+model_name)

        print()

    time_elapsed = time.time() - since
    df = pd.DataFrame({'losses': losses, 'accs': accs})
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc, df
	
	
	
	
#==============================================================================================
# для теста

def get_prediction(outputs):
    val, idx = torch.max(outputs, dim=1)
    return idx.data.cpu().view(-1)

def get_acc(preds, targets):
    return np.sum(preds==targets)/len(targets)

# выход: лосс, точность, список предсказанных моделью значений
def test_model(net, loader, criterion):
    net.eval()
    
    test_loss, test_acc = 0, 0
    print(test_loss, test_acc)
    count = 0
    # y_hat - список для результатов классификации
    y_hat = []
    
    for data, target in loader:
        count += 1
        
        if use_gpu:
            data = Variable(data.cuda(CUDA_DEVICE), volatile=True)
            target = Variable(target.cuda(CUDA_DEVICE))
        else:       
            data = Variable(data, volatile=True)
            target = Variable(target)
        
        output = net(data)
        #print(output.size(), target.size())
        test_loss += criterion(output, target).data[0]
        pred = (get_prediction(output)).numpy()
        y_hat.extend(pred)
        test_acc += get_acc(pred, target.data.cpu().numpy())
        if (count%10 == 0):
            print(count*len(data), end=" ")
            print(test_loss/(count*len(data)), test_acc/(count*len(data)))
        
    test_loss /= len(loader)
    test_acc /= len(loader)
    return test_loss, test_acc, y_hat