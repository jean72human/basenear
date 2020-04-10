
import torch
import numpy as np

import pickle
from autokeras.utils import ensure_dir, has_file, pickle_from_file, pickle_to_file

import os
import time

from autokeras.constant import Constant
from autokeras.search import BayesianSearcher, GreedySearcher, train

from autokeras.utils import pickle_to_file, rand_temp_folder_generator, ensure_dir
from autokeras.nn.generator import CnnGenerator, MlpGenerator, ResNetGenerator, DenseNetGenerator
from autokeras.nn.metric import Accuracy, MSE 
from autokeras.nn.loss_function import classification_loss, regression_loss

import torchvision

import neptune


from keras.datasets import cifar10

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()



    args = [0, len(x_train.shape) - 1] + list(range(1, len(x_train.shape) - 1))
    x_train = torch.Tensor(x_train.transpose(*args))
    x_test = torch.Tensor(x_test.transpose(*args))


    y_train = y_train.reshape(50000,)
    y_test = y_test.reshape(10000,)



    x_train = x_train/255
    x_test = x_test/255


    x_train = torch.Tensor(x_train) # transform to torch tensor
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test) 
    y_test = torch.Tensor(y_test)



    train_data = torch.utils.data.TensorDataset(x_train,y_train)
    test_data = torch.utils.data.TensorDataset(x_test,y_test)


    split = int(50000*0.08333)

    train_data, val_data = torch.utils.data.random_split(train_data,(50000-split,split))


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)


    searcher_args = {}
    input_shape = (32,32,3)
    n_classes = 10
    searcher_args['n_output_node'] = n_classes
    searcher_args['input_shape'] = input_shape
    searcher_args['path'] = ""
    searcher_args['metric'] = Accuracy
    searcher_args['loss'] = classification_loss
    #searcher_args['generators'] = [CnnGenerator, ResNetGenerator, DenseNetGenerator]
    searcher_args['generators'] = [CnnGenerator]
    searcher_args['verbose'] = True


    searcher = BayesianSearcher(**searcher_args)


    pickle.dump(searcher, open(os.path.join(searcher_args['path'], 'searcher'), 'wb'))


    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, 
                                            num_workers=0, shuffle=True)

    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        device = 'cuda'
        model.to(device)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.long().to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('Test loss: {:.9f}, Test Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        return test_loss, accuracy


    time_limit = 12 * 60 * 60 

    start_time = time.time()
    time_remain = time_limit

    neptune.init('jean72human/AutoMLCifar10',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYTM2NTU4OTUtZDNiNy00ZDM0LWE0MDUtZjNjMzZkOWU4ZWY0In0=')

    with neptune.create_experiment():
        while time_remain > 0:
            searcher.search(train_loader, val_loader, int(time_remain))
            time_elapsed = time.time() - start_time
            time_remain = time_limit - time_elapsed
            model = searcher.load_best_model().produce_model()
            t_loss, t_acc = test(model, test_loader)
            temp_time = (time.time() - start_time)/60
            neptune.send_metric('loss', t_loss)
            neptune.send_metric('accuracy', t_acc)
            neptune.send_metric('time', temp_time)




