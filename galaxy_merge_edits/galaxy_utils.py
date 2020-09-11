import numpy as np
import torch
import torch.nn as nn
import os.path
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import deque
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
        # print("Best loss: ", self.best_score)
        # print("New loss: ", score)
        # print("Counter: ", self.counter)
        # print("Patience: ", self.patience)

        self.meter.append(score)

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score:
            self.best_score = score
        elif score - self.best_score >= .4:
            self.counter += 1
        elif .3 <= score - self.best_score < .4:
            self.counter += .50
        elif .2 <= score - self.best_score < .3:
            self.counter += .25
        elif .1 <= score - self.best_score < .2:   
            self.counter += .1
        elif score - self.best_score <.1:
            self.counter = self.counter

        if self.counter > self.patience:
            print("counter is greater than patience")
            stop_sign = True

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

    # print("adnet output")
    # print(d_out)
    # print("batch size")
    # print(batch_size)
    # print("target")
    # print(d0_target)
    # print("prediction")
    # print(d0_pred)
    # print("all accuracy")
    # print(d0_acc)
    # print("source accuracy")
    # print(source_acc)
    # print("target accuracy")
    # print(target_acc)
    # sys.exit()
    
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

def distance_classification_test(loader, dictionary_val, model, centroids, gpu=True, verbose = False, save_where = None):
    start_test = True
    with torch.no_grad():
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

    if verbose:

        output = pd.DataFrame()

        print("all output")
        print(all_output.cpu().detach().numpy())
        print(torch.max(all_output, 1)[1].cpu().detach().numpy())
        print()
        print("all label")
        print(all_label.cpu().detach().numpy())

        output['model output'] = pd.Series(torch.max(all_output, 1)[1].cpu().detach().numpy())
        output['labels'] = pd.Series(all_label.cpu().detach().numpy())

        output.to_csv(str(save_where)+"/model_results.csv")

    return accuracy, conf_matrix

def image_classification_test(loader, dictionary_val, model, gpu=True, verbose = False, save_where = None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader[str(dictionary_val)])
        for i in range(len(loader[str(dictionary_val)])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(1.0 * outputs)

            if start_test:
                all_softmax_output = softmax_outputs.data.float()
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.float()), 0)
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

    #_, predict = torch.max(all_output, 1)
    _, predict = torch.max(all_softmax_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).float() / float(all_label.size()[0])
    conf_matrix = confusion_matrix(all_label.cpu().numpy(), predict.cpu().numpy())

    if verbose:

        output = pd.DataFrame()

        df = pd.DataFrame(all_softmax_output.cpu().detach().numpy(), columns=['non-merger', 'merger'])

        # print("all output")
        # print(all_output.cpu().detach().numpy())
        # print(torch.max(all_output, 1)[1].cpu().detach().numpy())
        # print()
        # print("all label")
        # print(all_label.cpu().detach().numpy())

        output['model output'] = pd.Series(torch.max(all_output, 1)[1].cpu().detach().numpy())
        output['labels'] = pd.Series(all_label.cpu().detach().numpy())

        output.to_csv(str(save_where)+"/model_results.csv")
        df.to_csv(str(save_where)+"/model_predictions.csv")

    return accuracy, conf_matrix

def image_classification_predict(loader, dictionary_val, model, gpu=True, softmax_param=1.0):
    start_test = True
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


def plot_embedding(X, y, d, title=None, imgName=None, save_dir=None):
    """
    Plot an embedding X with the class label y colored by the domain d.
    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image
    :return:
    """
    fig_mode = 'save'

    if fig_mode is None:
        return

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    alpha_list = [.3, 1]

    #set opacity for domain> [source, target]
    alpha_list = [.3, 1]

    for i in range(X.shape[0]):
      
    	plt.scatter(X[i, 0], X[i, 1], marker='o', alpha= alpha_list[d[i]],
              color=plt.cm.bwr(y[i]/1.))

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title('Blobs')

    if fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()

    if fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        print('Saving ' + imgName + ' ...')
        plt.savefig(imgName)
        plt.close()



def visualizePerformance(base_network, src_test_dataloader,
                         tgt_test_dataloader, batch_size, domain_classifier=None, num_of_samples=None, imgName=None, use_gpu=True, save_dir=None):
    """
    Evaluate the performance of dann and source only by visualization.
    :param feature_extractor: network used to extract feature from target samples
    #:param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :batch_size: batch size used in the main code 
    :param num_of_samples: the number of samples (from train and test respectively) for t-sne
    :param imgName: the name of saving image

    :return:
    """

    # Setup the network
    #feature_extractor.eval()
    #class_classifier.eval()
    base_network.eval()
    if domain_classifier is not None:
        domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    if num_of_samples is None:
        num_of_samples = batch_size
    else:
        assert len(src_test_dataloader) * num_of_samples, \
            'The number of samples can not bigger than dataset.' # NOT PRECISELY COMPUTATION

    # Collect source data.-- labeled with 0
    s_images, s_labels, s_tags = [], [], []
    for batch in src_test_dataloader:
        images, labels = batch

        if use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)
        
        s_labels.append(labels)
        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        if len(s_images * batch_size) > num_of_samples:
            break

    s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], \
                                 torch.cat(s_labels)[:num_of_samples], torch.cat(s_tags)[:num_of_samples]

    # Collect test data.-- labeled with 1
    t_images, t_labels, t_tags = [], [], []
    for batch in tgt_test_dataloader:
        images, labels = batch

        if use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)

        t_labels.append(labels)
        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))

        if len(t_images * batch_size) > num_of_samples:
            break

    t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], \
                                 torch.cat(t_labels)[:num_of_samples], torch.cat(t_tags)[:num_of_samples]

    # Compute the embedding of target domain.
    embedding1, logits = base_network(s_images)
    embedding2, logits = base_network(t_images)

    tsne = TSNE(perplexity=100, metric= 'cosine', n_components=2, init='pca', n_iter=3000)

    if use_gpu:
        network_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))
    else:
        network_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                   embedding2.detach().numpy())))


    plot_embedding(network_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), 'tSNE', imgName, save_dir)
