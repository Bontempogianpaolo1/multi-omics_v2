from itertools import cycle, islice

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
import itertools
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
from venn import venn
from sklearn.manifold import TSNE
import umap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seed = 1200


def plot_roc(n_classes, y_score, y_test2, names, title):
    y_score = pd.factorize(y_score, )
    y_test2 = pd.factorize(y_test2, sort=True)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes - 1):
        fpr[i], tpr[i], _ = roc_curve(y_test2[0], y_score[0], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area

    # Finally average it and compute AUC

    lw = 2
    # PlotDir all ROC curve
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i, color in zip(range(n_classes - 1), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.pause(0.2)


def plot_pareto(max_variance, variance, n_components, title):
    fig, ax = plt.subplots()
    max_variance = variance.max()
    # variance=variance/max_variance*100
    ax.bar(n_components, variance, color="C0")
    ax2 = ax.twinx()
    ax2.plot(n_components, variance.cumsum() / variance.sum(), color="C1", marker="D")
    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax.tick_params(axis="y", colors="C0")
    ax.set_ylabel('variance(variance / best) ')
    ax.set_xlabel('number of principal components')
    ax2.tick_params(axis="y", colors="C1")
    ax2.set_ylabel('variance cumulative')
    plt.title(title)
    plt.savefig("../Data/outputs/paretos/" + title + ".png")
    plt.show()


def pareto_plot(df, title=None, show_pct_y=False, pct_format='{:.1%}'):
    """
           df = pd.DataFrame({
               'components': components[1:].tolist(),
               'variance': variances[1:].tolist(),
               'score': scores[1:].tolist(),
           })
    """
    tmp = df
    x = tmp['components'].values
    variances = tmp['variance'].values
    scores = tmp['score'].values
    weights = variances / variances.sum()
    cumsum = weights.cumsum()

    fig, ax1 = plt.subplots()
    ax1.bar(x, variances, color='dodgerblue')
    ax1.set_xlabel("pc")
    ax1.set_ylabel("variance", color='dodgerblue')
    ax1.tick_params('y', colors='dodgerblue')

    ax2 = ax1.twinx()
    ax2.plot(x, scores, '-ro', alpha=0.5)
    ax2.set_ylabel('accuracy', color='r')
    ax2.tick_params('y', colors='r')
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    # ax2.yaxis.set_major_formatter(PercentFormatter())
    # hide y-labels on right side
    # if not show_pct_y:
    #    ax2.set_yticks([])
    formatted_weights = [pct_format.format(x) for x in scores]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], scores[i]), fontweight='heavy')

    ax3 = ax1.twinx()
    offset = 60
    ax3.spines["right"].set_position(("axes", 1.2))

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    make_patch_spines_invisible(ax3)
    # Second, show the right spine.
    ax3.spines["right"].set_visible(True)
    ax3.plot(x, cumsum, '-go', alpha=0.5)
    ax3.set_ylabel('cumulative variance', color='g')
    ax3.tick_params('y', colors='g')
    vals = ax3.get_yticks()
    ax3.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    # hide y-labels on right side
    # ax3.yaxis.set_major_formatter(PercentFormatter())
    # if not show_pct_y:
    #    ax3.set_yticks([])
    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax3.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')

    # if title:
    #    plt.title(title)

    plt.tight_layout()
    plt.savefig("../Data/outputs/paretos/" + title + ".png")
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("../Data/outputs/" + title + ".png")
    plt.pause(0.2)


# Learning curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.pause(0.2)


def plot_diagram(df):
    # number of elements per class
    counts = df.iloc[:, 0].value_counts()
    # different classes
    names = counts.index
    # tot classes
    x = np.arange(counts.shape[0])
    plt.close()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    plt.title("Elements per class")
    colors = np.array(list(islice(cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k']), int(counts.shape[0] + 1))))
    ax1.barh(x, counts, 0.75, color=colors)
    ax1.set_yticks(x + 0.75 / 2)
    ax1.set_xticks([])
    ax1.set_yticklabels(names, minor=False)
    wedges, texts, autotexts = ax2.pie(counts, autopct=lambda pct: "{:.1f}%\n".format(pct))
    ax2.legend(wedges, names, title="Ingredients", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    for i, v in enumerate(counts):
        ax1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    plt.pause(0.2)
    print(counts)
    print("diagram displayed")
    print("===================")


def plot_outliers(X, y, X_train, title):
    # pca=PCA(n_components=3)
    # x_train=pca.fit_transform(x_train)
    # X=pca.transform(X)
    # X=pca.transform(X)
    # x_train=pca.transform(x_train)
    # x_train=x_train.to_numpy()
    colors = np.array(list(islice(cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k']), 4)))
    markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), 4)))
    lista = [1, -1]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    for l, c, m in zip(lista, colors, markers):
        ax.scatter(X[y == l, 0], X[y == l, 1], X[y == l, 2], c=c, marker=m, label='class %s' % l)

    colors = np.array(list(islice(cycle(['g', 'c', 'm', 'y', 'k', 'b', 'r']), 4)))
    markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), 4)))
    lista = [1]
    for l, c, m in zip(lista, colors, markers):
        ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=c, marker=m, label='train-set without outliers')
    ax.legend(loc='upper left', fontsize=12)
    plt.title(title)
    plt.savefig("../Data/outputs/" + title + ".png")
    plt.pause(0.2)


def plot_sets(multi_names, multi_values, data, filenames, modelname, multi_y_pred, multi_y_true):
    from matplotlib.pyplot import subplots
    from itertools import chain, islice
    from string import ascii_uppercase
    from numpy.random import choice
    official_name0 = data[0]["official_name"][data[0]["y_pred"] != data[0]["y_true"]].to_numpy()
    official_name1 = data[1]["official_name"][data[1]["y_pred"] != data[1]["y_true"]].to_numpy()
    official_name2 = data[2]["official_name"][data[2]["y_pred"] != data[2]["y_true"]].to_numpy()
    # defining sets for both subjects
    # venn3([set(official_name0.flatten()),set(official_name1.flatten()),set(official_name2.flatten()),set(multi_names.to_numpy().flatten())],set_labels=[1,2,3,4])
    result = {
        "multi": set(multi_names.to_numpy().flatten()),
        filenames[0]: set(official_name0.flatten()),
        filenames[1]: set(official_name1.flatten()),
        filenames[2]: set(official_name2.flatten()),
    }
    result2 = result.copy()
    result2.get("multi").intersection(result2.get(filenames[0]))
    with open("../Data/outputs/without-unknown-testset-withweigths-" + modelname + "-probabilities.txt", 'w') as f:
        for name1, name2, name3 in itertools.combinations_with_replacement(filenames, 3):
            title = "relazione multi-" + name1 + "-" + name2 + "-" + name3 + "-modello-" + modelname
            print(title, file=f)
            intersection = result2.get("multi").intersection(result2.get(name1)) \
                .intersection(result2.get(name2)) \
                .intersection(result2.get(name3))
            print("totali:" + str(len(intersection)), file=f)
            print(file=f)
            print("id:", file=f)
            print(np.array(list(intersection)), file=f)
            print(file=f)

            prob_multi2 = multi_values[
                [False if item not in np.array(list(intersection)) else True for item in multi_names]]
            multi_y_pred2 = multi_y_pred[
                [False if item not in np.array(list(intersection)) else True for item in multi_names]]
            multi_y_true2 = multi_y_true[
                [False if item not in np.array(list(intersection)) else True for item in multi_names]]
            plot_misclassified(np.array(list(intersection)), multi_y_pred2, "multi-omica prediction", title)
            print("probabilita multi omica", file=f)
            print(prob_multi2.numpy(), file=f)
            print("pred multi omica", file=f)
            print(multi_y_pred2.numpy(), file=f)
            print("true multi omica", file=f)
            print(multi_y_true2.values, file=f)
            print(file=f)
            for name in [name1, name2, name3]:
                print("probabilità " + name, file=f)
                probability = data[filenames.index(name)]["max_probability"][
                    [False if item not in np.array(list(intersection)) else True for item in
                     data[filenames.index(name)]["official_name"]]]
                y_pred = data[filenames.index(name)]["y_pred"][
                    [False if item not in np.array(list(intersection)) else True for item in
                     data[filenames.index(name)]["official_name"]]]
                y_true = data[filenames.index(name)]["y_true"][
                    [False if item not in np.array(list(intersection)) else True for item in
                     data[filenames.index(name)]["official_name"]]]
                # plot_misclassified(np.array(list(intersection)), y_pred, name, title)
                print(probability.values, file=f)
                print("y_pred " + name, file=f)
                print(y_pred.values, file=f)
                print("y_true " + name, file=f)
                print(y_true.values, file=f)
                print(file=f)

    _, top_axs = subplots(ncols=1, nrows=1, figsize=(5, 5))
    # _, bot_axs = subplots(ncols=2, nrows=1, figsize=(18, 8))
    cmaps = [list("rgbm"), "plasma", "viridis", "Set1"]
    letters = iter(ascii_uppercase)

    # for n_sets, cmap in zip(range(2, 7), cmaps):
    #    dataset_dict = {
    #        name: set(choice(1000, 700, replace=False))
    #        for name in islice(letters, n_sets)
    #    }
    venn(result, cmap=cmaps[0], fontsize=8, legend_loc="upper left", ax=top_axs)
    # venn(result)
    plt.plot()
    title = "set-theory" + modelname
    plt.savefig("../Data/outputs/" + title + ".png")
    plt.pause(0.2)
    return


def plot_misclassified(intersection, y_pred, name, title):
    annotation_path = "../Data/data/kidney/preprocessed_annotation_global.csv"
    y = pd.read_csv(annotation_path)
    # y = pd.read_csv(annotation_path)["label"]
    # names = y.astype('category').cat.categories
    # y = y.astype('category').cat.codes
    modelname = " mlp "
    meth_path = "../Data/data/kidney/preprocessed_Matrix_meth.csv"
    mRNA_path = "../Data/data/kidney/preprocessed_Matrix_miRNA_deseq_correct.csv"
    mRNA_normalized_path = "../Data/data/kidney/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
    files = [meth_path, mRNA_normalized_path, mRNA_path]
    filenames = ["meth", "mRNA", "miRNA"]
    parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
                  'max_iter': [200, 400, 600]}
    true_labels = []
    for file, filename in zip(files, filenames):
        title2 = title + "-" + filename
        outputname = modelname + filename
        X = pd.read_csv(file, index_col=False, header=None)
        if filename == "miRNA":
            X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed,
                                                            stratify=y["label"].astype('category').cat.codes)

        # pca=TSNE(n_components=2)

        pca = umap.UMAP()

        pca.fit_transform(X_train)
        X = pca.transform(X)
        y2 = y["label"].astype('category').cat.codes
        y2[[False if item not in intersection else True for item in y["official_name"]]] = 5
        print(y2[y2 == 5].shape)
        colors = np.array(list(islice(cycle(['b', 'm', 'g', 'c', 'y', 'r', 'k']), 6)))
        markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), 6)))
        lista = [0, 1, 2, 3, 4, 5, 6]

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()
        for l, c, m in zip(lista, colors, markers):
            ax.scatter(X[y2 == l, 0], X[y2 == l, 1], c=c, marker=m, label='class %s' % l)

        colors = np.array(list(islice(cycle(['g', 'c', 'm', 'y', 'k', 'b', 'r']), 4)))
        markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), 4)))
        lista = [1]
        # for l, c, m in zip(lista, colors, markers):
        #    ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=c, marker=m, label='train-set without outliers')
        ax.legend(loc='upper left', fontsize=12)
        plt.title(title2)
        plt.savefig("../Data/outputs/UMAP-2d-" + title2 + ".png")
        plt.pause(0.2)
    return


def PlotInstograms(df: pd.DataFrame,title):
    df2=df.copy()
    df2['max_probability']=round(df['max_probability'],2)
    sns.catplot(x="max_probability", kind="count", data=df2)
    plt.title(title)
    plt.savefig("../Data/outputs/" + title + ".png")
    plt.show()

