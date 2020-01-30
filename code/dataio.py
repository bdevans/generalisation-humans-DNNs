import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mappings import HumanCategories


# train_images = '/work/data/imagenet_2012/'  # 1281167
# train_16_images = '/work/data/16-class-ImageNet/images'
# val_images = '/work/data/imagenet_2012_val/'  # 50000
# # test_images = '/work/data/imagenet_2012_test/'  # 100000 - No labels!
# output_dir = '/work/data/16-class-ImageNet/'

hc = HumanCategories()

# df_full = pd.read_csv(os.path.join(train_images, 'train.txt'), delimiter=' ', header=None, names=['filename', 'class']) 

# # df_test = pd.read_fwf(os.path.join(test_images, 'test.txt'), header=None, names=['filename', 'class'])

# df_full['wnid'] = df_full['class'].map(hc.ind1000_wnid)
# df_full['label'] = df_full['class'].map(hc.ind1000_label)
# df_full['ind16'] = df_full['class'].map(hc.ind1000_ind16)
# df_full['hypernym'] = df_full['class'].map(hc.ind1000_hyp)
# df_full_clean = df_full.copy().dropna().astype({'class': 'uint16', 'ind16': 'uint8', 'hypernym': 'category'})  # 'class': 'U'
# train_weights = dict(1/df_full_clean['hypernym'].value_counts())
# df_full_clean['weights'] = df_full_clean['hypernym'].map(train_weights)

# df_full.sample(n=10)


def get_df_train(image_lists='/work/data/16-class-ImageNet/image_names', output_dir=None):
    """Construct a dataframe for the 16 class ImageNet dataset with class weights and labels"""
    dfs = []
    n_classes = len(hc.get_hypernyms())
    for h, hypernym in enumerate(hc.get_hypernyms()):
        df = pd.read_csv(os.path.join(image_lists, hypernym+'.txt'), 
                         header=None, names=['filename'])
        df['filename'] = hypernym + os.path.sep + df['filename']
        df['hypernym'] = hypernym
        df['ind16'] = h
        df['class16'] = f"{h:02d}"
        df['weights'] = 1/len(df)/n_classes  # All weights for all classes sum to 1
        dfs.append(df)
    df_train = pd.concat(dfs)
    df_train = df_train.astype({'class16': str})  # Ensure this is a string for correct ordering
    # Weight is "proportional to" 1 / number of samples in class
    df_train['weights'] = df_train['weights'].multiply(len(df_train))
    if output_dir:
        df_train.to_csv(os.path.join(output_dir, 'train_16.csv'))  # 213,555
    return df_train


def get_df_train_full(image_dir = '/work/data/imagenet_2012/', output_dir=None):
    df_full = pd.read_csv(os.path.join(train_images, 'train.txt'), delimiter=' ', 
                          header=None, names=['filename', 'class'])
    df_full['wnid'] = df_full['class'].map(hc.ind1000_wnid)
    df_full['label'] = df_full['class'].map(hc.ind1000_label)
    df_full['ind16'] = df_full['class'].map(hc.ind1000_ind16)
    df_full['hypernym'] = df_full['class'].map(hc.ind1000_hyp)
    if output_dir:
        df_full.to_csv(os.path.join(output_dir, 'train_full.csv'))  # 1,281,167
    df_full_clean = df_full.copy().dropna().astype({'class': 'uint16', 'ind16': 'uint8', 'hypernym': 'category'})  # 'class': 'U'
    train_weights = dict(1/df_full_clean['hypernym'].value_counts())
    df_full_clean['weights'] = df_full_clean['hypernym'].map(train_weights)
    if output_dir:
        # This approximates the 16 class image set used by Geirhos et al.
        df_full_clean.to_csv(os.path.join(output_dir, 'train_16.csv'))  # 262,368
    return


def get_df_valid(image_dir='/work/data/imagenet_2012_val/', output_dir=None):
    # df_val = pd.read_fwf(os.path.join(image_dir, 'val.txt'), 
    #                      header=None, names=['filename', 'class'])
    df_val = pd.read_csv(os.path.join(image_dir, 'val.txt'), delimiter=' ',
                         header=None, names=['filename', 'class'])
    df_val['wnid'] = df_val['class'].map(hc.ind1000_wnid)
    df_val['label'] = df_val['class'].map(hc.ind1000_label)
    df_val['ind16'] = df_val['class'].map(hc.ind1000_ind16)
    df_val['hypernym'] = df_val['class'].map(hc.ind1000_hyp)
    if output_dir:
        df_val.to_csv(os.path.join(output_dir, 'valid_full.csv'))
    df_val_clean = df_val.copy().dropna().astype({'class': 'uint16', 'ind16': 'uint8', 'hypernym': 'category'})
    test_weights = dict(1/df_val_clean['hypernym'].value_counts())
    df_val_clean['weights'] = df_val_clean['hypernym'].map(test_weights)
    # Pad ind16 with zeros and save as a str for correct (alphabetical) ordering
    df_val_clean['class16'] = df_val_clean['ind16'].apply(lambda x: f"{x:02d}")
    df_val_clean = df_val_clean.astype({'class16': str})
    if output_dir:
        df_val_clean.to_csv(os.path.join(output_dir, 'valid_16.csv'))  # 50000
    return df_val_clean


_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def decode_predictions_16(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.
    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    # global CLASS_INDEX
    CLASS_INDEX = hc.CLASS_INDEX_16

    backend, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    if len(preds.shape) != 2 or preds.shape[1] != 16:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 16)). '
                         'Found array with shape: ' + str(preds.shape))
    # if CLASS_INDEX is None:
    #     fpath = keras_utils.get_file(
    #         'imagenet_class_index.json',
    #         CLASS_INDEX_PATH,
    #         cache_subdir='models',
    #         file_hash='c2c37ea517e94d9795004a39431a14cb')
    #     with open(fpath) as f:
    #         CLASS_INDEX = json.load(f)

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def plot_predictions(model, decoder, images, labels=None, top=5, 
                     wrap=4, horizontal=True, subfig_size=3):
    # Get predictions for a batch

#     print(images.shape)
    batch_size, rows, cols, n_channels = images.shape
    yhat = model.predict(images)
    n_classes = yhat.shape[1]
#     print(yhat.shape)
#     pprint.pprint(yhat)
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
    predictions = decoder(yhat, top=top)  # decode_predictions
#     print(len(predictions))
    if horizontal:
        nrows, ncols = 2 * int(np.ceil(len(predictions) / wrap)), wrap
    else:
        nrows, ncols = wrap, 2 * int(np.ceil(len(predictions) / wrap))
    # fig, axes = plt.subplots(nrows=len(predictions), ncols=2, figsize=(3*2, 3*batch_size))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subfig_size*ncols, subfig_size*nrows))
    for i, image in enumerate(images):
        if horizontal:
            row, col = 2 * (i // ncols), i % ncols
        else:
            row, col = i % nrows, 2 * (i // nrows)
        # Plot original image
        axes[row, col].imshow(image, aspect='auto')
        axes[row, col].set_axis_off()
        if labels is not None:
            label = decoder(labels[np.newaxis, i, :])[0][0][1]
            axes[row, col].set_title(label)
        # Plot predictions
#         scores = [prediction[2] for prediction in predictions[i]]
#         print(scores)
        if horizontal:
            row += 1
        else:
            col += 1
        bars = axes[row, col].bar(range(top), [prediction[2] for prediction in predictions[i]])
        if n_classes == 1000:
            xticklabels = ['\n'.join(prediction[:-1]) for prediction in predictions[i]]
        elif n_classes == 16:
            xticklabels = [prediction[1] for prediction in predictions[i]]
        else:
            raise ValueError(f"Unexpected number of classes: {n_classes}!")
        axes[row, col].set_xticks(range(top))
        axes[row, col].set_xticklabels(xticklabels)
        axes[row, col].set_ylim((0, 1))

        for b, (bar, prediction) in enumerate(zip(bars, predictions[i])):
            axes[row, col].annotate(f"{prediction[2]:.1%}", 
                                    xy=(bar.get_x(), bar.get_height()), 
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
    return fig, axes


def check_image(image):
    # assert image.dtype in [np.float32, np.float64]
    print(f"Type: {type(image)}")
    print(f"Shape: {image.shape}")
    print(f"Pixel intensity: [{np.amin(images)}, {np.amax(images)}]")

