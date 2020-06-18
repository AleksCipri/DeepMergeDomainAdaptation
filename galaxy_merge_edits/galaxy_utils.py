import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from collections import deque


class EarlyStopping(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events
    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
    """
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.meter = deque(maxlen=patience)

    def is_stop_training(self, score):
        stop_sign = False
        self.meter.append(score)
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                stop_sign = True
        # approximately equal
        elif np.abs(score.cpu() - self.best_score) < 1e-9:
            if len(self.meter) == self.patience and np.abs(np.mean([x.item() for x in self.meter]) - score.cpu()) < 1e-7:
                stop_sign = True
            else:
                self.best_score = score.cpu()
                self.counter = 0
        else:
            self.best_score = score.cpu()
            self.counter = 0
        return stop_sign


def domain_cls_accuracy(d_out):
    '''domain classification accuracy
    Args: 
        d_out: torch.FloatTensor, output of the domain classification network
    Returns:
        d0_acc: float, domain classification accuracy of both domains
        source_acc: float, domain classification accuracy of the source domain
        target_acc: float, domain classification accuracy of the target domain
    '''
    batch_size = d_out.size(0) // 2
    d0_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    d0_pred = d_out.detach() >= 0.5
    d0_acc = float(d0_pred.data.cpu().float().eq(d0_target).sum()) / float(d_out.size(0))
    source_acc = float(d0_pred[:batch_size].cpu().float().eq(d0_target[:batch_size]).sum()) / float(batch_size)
    target_acc = float(d0_pred[batch_size:].cpu().float().eq(d0_target[batch_size:]).sum()) / float(batch_size)
    return d0_acc, source_acc, target_acc


def distance_to_centroids(x, centroids):
    '''euclidean distance of a batch of samples to class centers
    Args:
        x: FloatTensor [batch_size, d]
        centroids: FloatTensor [K, d] where K is the number of classes
    Returns:
        dist: FloatTensor [batch_size, K]
    '''
    b, d = x.size()
    K, _ = centroids.size()
    dist = x.unsqueeze(1).expand(b, K, d) - centroids.unsqueeze(0).expand(b, K, d)
    return torch.norm(dist, dim=-1)


def distance_classification_test(loader, dictionary_val, model, centroids, gpu=True):
    start_test = True
    with torch.no_grad():
        # if test_10crop:
        #     iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        #     for i in range(len(loader['test0'])):
        #         data = [iter_test[j].next() for j in range(10)]
        #         inputs = [data[j][0] for j in range(10)]
        #         labels = data[0][1]
        #         if gpu:
        #             for j in range(10):
        #                 inputs[j] = inputs[j].cuda()
        #             labels = labels.cuda()
        #             centroids = centroids.cuda()
        #         outputs = []
        #         for j in range(10):
        #             features, _ = model(inputs[j])
        #             dist = distance_to_centroids(features, centroids)
        #             outputs.append(nn.Softmax(dim=1)(-1.0 * dist))
        #         outputs = sum(outputs)
        #         if start_test:
        #             all_output = outputs.data.float()
        #             all_label = labels.data.float()
        #             start_test = False
        #         else:
        #             all_output = torch.cat((all_output, outputs.data.float()), 0)
        #             all_label = torch.cat((all_label, labels.data.float()), 0)
        # else:
        iter_test = iter(loader[str(dictionary_val)])
        for i in range(len(loader[str(dictionary_val)])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
                centroids = centroids.cuda()
            features, _ = model(inputs)
            dist = distance_to_centroids(features, centroids)
            outputs = nn.Softmax(dim=1)(-1.0 * dist)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)       
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).float() / float(all_label.size()[0])
    conf_matrix = confusion_matrix(all_label.cpu().numpy(), predict.cpu().numpy())
    return accuracy, conf_matrix

def image_classification_test(loader, dictionary_val, model, gpu=True):
    start_test = True
    with torch.no_grad():
        # if test_10crop:
        #     iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        #     for i in range(len(loader['test0'])):
        #         data = [iter_test[j].next() for j in range(10)]
        #         inputs = [data[j][0] for j in range(10)]
        #         labels = data[0][1]
        #         if gpu:
        #             for j in range(10):
        #                 inputs[j] = inputs[j].cuda()
        #             labels = labels.cuda()
        #         outputs = []
        #         for j in range(10):
        #             _, predict_out = model(inputs[j])
        #             outputs.append(nn.Softmax(dim=1)(predict_out))
        #         outputs = sum(outputs)
        #         if start_test:
        #             all_output = outputs.data.float()
        #             all_label = labels.data.float()
        #             start_test = False
        #         else:
        #             all_output = torch.cat((all_output, outputs.data.float()), 0)
        #             all_label = torch.cat((all_label, labels.data.float()), 0)
        # else:
        iter_test = iter(loader[str(dictionary_val)])
        for i in range(len(loader[str(dictionary_val)])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).float() / float(all_label.size()[0])
    conf_matrix = confusion_matrix(all_label.cpu().numpy(), predict.cpu().numpy())
    return accuracy, conf_matrix

def image_classification_predict(loader, dictionary_val, model, gpu=True, softmax_param=1.0):
    start_test = True
    # if test_10crop:
    #     iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
    #     for i in range(len(loader['test0'])):
    #         data = [iter_test[j].next() for j in range(10)]
    #         inputs = [data[j][0] for j in range(10)]
    #         labels = data[0][1]
    #         if gpu:
    #             for j in range(10):
    #                 inputs[j] = Variable(inputs[j].cuda())
    #             labels = Variable(labels.cuda())
    #         else:
    #             for j in range(10):
    #                 inputs[j] = Variable(inputs[j])
    #             labels = Variable(labels)
    #         outputs = []
    #         for j in range(10):
    #             _, predict_out = model(inputs[j])
    #             outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
    #         softmax_outputs = sum(outputs)
    #         if start_test:
    #             all_softmax_output = softmax_outputs.data.cpu().float()
    #             start_test = False
    #         else:
    #             all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    # else:
    iter_val = iter(loader[str(dictionary_val)])
    for i in range(len(loader[str(dictionary_val)])):
        data = iter_val.next()
        inputs = data[0]
        if gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, outputs = model(inputs)
        softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
        if start_test:
            all_softmax_output = softmax_outputs.data.cpu().float()
            start_test = False
        else:
            all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    return all_softmax_output