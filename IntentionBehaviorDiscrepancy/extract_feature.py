import os
import pickle
import keras
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

from layers import CoAttentionParallel
from sklearn.cluster import DBSCAN
from sklearn import metrics

# =========================
# load model and data
# =========================

def save_pkl_data(path, data):
    with open(path, 'wb') as fo:
        pickle.dump(data, fo)

def load_pkl_data(path):
    with open(path, 'rb') as fi:
        data = pickle.load(fi)
    return data

def load_model(path_model):
    model = keras.models.load_model(path_model, custom_objects={
        'CoAttentionParallel': CoAttentionParallel
    })
    extract_f = keras.models.Model(
        inputs=model.input, outputs=model.get_layer('feature').output
    )
    extract_p = model

    return extract_f, extract_p

def load_meta(path_model_meta):
    meta = load_pkl_data(path_model_meta)

    image_shape = meta['ModelConf'].img_shape
    re_sample_type = meta['ModelConf'].img_re_sample
    text_len = meta['ModelConf'].text_length
    id2label = {v: k for k, v in meta['label2id'].items()}
    permission_names = [id2label[i] for i in range(len(id2label))]

    return image_shape, re_sample_type, text_len, permission_names

# =========================
# process input data
# =========================
def process_input_data(inputs_raw, img_shape, re_sample_type, text_len, permission_names):
    img_size, img_channel = img_shape[:2], img_shape[-1]

    input_images, input_texts, permissions = [], [], []
    for img_data, tokens, permission_label in inputs_raw:
        input_images.append(prepare_image(img_data, img_size, img_channel, re_sample_type))
        input_texts.append(prepare_text(tokens, text_len))
        permissions.append(prepare_permissions(permission_label, permission_names))

    input_images = np.array(input_images)
    input_texts = np.array(input_texts)
    inputs = [input_images, input_texts]

    return inputs, permissions

def prepare_image(img_data, target_size, target_channel, re_sample_type):
    # resize the image
    img = image_decompress(*img_data)
    img = img.resize(target_size, re_sample_type)

    # convert image
    if target_channel == 4:
        img = img.convert('RGBA')
    elif target_channel == 3:
        img = img.convert('RGB')
    elif target_channel == 1:
        img = img.convert('L')

    # transform to 0.0 to 1.0 values
    np_img = np.array(img) / 255.0

    return np_img

def image_decompress(img_mode, img_size, img):
    img = Image.frombytes(img_mode, img_size, img)

    return img

def prepare_text(words, max_len):
    if len(words) < max_len:
        words = words + [0] * (max_len - len(words))
    else:
        words = words[:max_len]
    return np.array(words)

def prepare_permissions(label, permission_names):
    return {permission_names[p] for p in label}   

def prepare_input_data(data_orig, img_shape, re_sample_type, text_len, permission_names):
    inputs_raw, outlier_labels, id_labels = [], [], []
    for i in range(len(data_orig)):
        id_labels.append(data_orig[i][-1])
        outlier_labels.append(data_orig[i][-2])
        inputs_raw.append(data_orig[i][:3])
    inputs, permissions = process_input_data(
        inputs_raw, img_shape, re_sample_type, text_len, permission_names
    )
    return inputs, permissions, outlier_labels, id_labels

# =========================
# feature extraction
# =========================

def get_features(inputs, extract_f):
    features = extract_f.predict(inputs)
    return features
    


def prepare_apks():
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data', 'total')

    path_meta = os.path.join(path_data, 'deepintent.meta')
    path_model = os.path.join(path_data, 'deepintent.model')
    path_input = os.path.join(path_data, 'outlier.combined.pkl')
    path_out = os.path.join(path_data, 'outlier.processed.pkl')
    #load DeepIntent's models
    extract_f, extract_p = load_model(path_model)
    image_shape, re_sample_type, text_len, permission_names = load_meta(
        path_meta
    )
    #load our combined pkl file
    _, data_orig = load_pkl_data(path_input)  #format: [v2id, l2id], [[(img_data),[tokens], [perms], [outlier_labels], id], ...]

    print("pre-processing...")
    processed_input, permissions, outlier_labels, id_labels = prepare_input_data(data_orig, image_shape, re_sample_type, text_len, permission_names)
    print("extracting features...")
    extracted_features = get_features(processed_input, extract_f)
    #save features to a pkl file
    save_pkl_data(path_out, [extracted_features, permissions, outlier_labels, id_labels])



def main():
    prepare_apks()


if __name__ == '__main__':
    main()
