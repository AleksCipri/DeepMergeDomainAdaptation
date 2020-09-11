import os
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import scikitplot as skplt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from sklearn.preprocessing import label_binarize


#%%
#load from directory folder
os.chdir('C:/Users/dkafkes/Desktop/fermi/high velocity AI')
acc = EventAccumulator("missed-before")
acc.Reload()

# Print tags of contained entities, use these names to retrieve entities as below
params = acc.Tags()['scalars']
#%%

#plot with training+validation
training_accuracy = [(s.step, 100*s.value) for s in acc.Scalars('training_accuracy')]
validation_accuracy = [(s.step, 100*s.value) for s in acc.Scalars('validation_accuracy')]
training_loss = [(s.step, s.value) for s in acc.Scalars('training_total_loss')]
validation_loss = [(s.step, s.value) for s in acc.Scalars('validation_total_loss')]


param_list = [training_accuracy, validation_accuracy, training_loss, validation_loss]
for_plot = ['training acc %', 'validation acc %', 'training loss', 'validation loss']

fig = plt.figure()
ax = plt.subplot(111)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
plt.title("Training vs. Validation")
plt.xlabel("Epoch")
plt.ylabel("Value")

for i in range(0, len(param_list)):
    x, y = zip(*(param_list[i]))
    plt.plot(x,y, label=for_plot[i])

plt.ylim(0,100)
plt.yticks(np.arange(0, 100, step=10))
ax.legend(loc='left', bbox_to_anchor=(.6, .5),
          ncol=1)
plt.show()

#%%
#plot side by side deepmerge and resnet ROC curves for source, target all 3 runs
#am outputting in csv file


os.chdir('C:/Users/dkafkes/Desktop/fermi/high velocity AI/missed-before')
predictions = pd.read_csv('model_predictions.csv')[['non-merger', 'merger']]
ground_truth = pd.read_csv('model_results.csv')[['model output', 'labels']]

#%%
y_true = np.asarray(ground_truth.labels)
y_probas = np.asarray(predictions)
plot_roc2(y_true, y_probas, plot_micro = False, plot_macro = False) #classes_to_plot=['non-mergers', 'mergers'])
plt.show()

#%%

a = np.in1d(np.unique(ground_truth), [[0, 'nonmerger'], [1,'merger']])
print(a)

#%%
#adversarial plot with source and domain accuracy

#%%
def plot_roc2(y_true, y_probas, title='ROC Curves',
                   plot_micro=True, plot_macro=True, classes_to_plot=None,
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".
        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.
        plot_macro (boolean, optional): Plot the macro average ROC curve.
            Defaults to ``True``.
        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = ['non-merger', 'merger']
        classes_to_plot2 = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot2)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i],
                                                pos_label=classes_to_plot[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                          ''.format(classes[i], roc_auc))

    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='deeppink', linestyle=':', linewidth=4)

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr,
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax