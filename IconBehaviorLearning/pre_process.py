# -*- coding: UTF-8 -*-

"""Pre-process the raw data.

The pre-processing phase will 1) check the image, 2) tokenize and
stem texts, 3) extract sensitive permissions and map permissions to
permission groups.
"""

import os
from collections import Counter

import nltk
import autocorrect

from tools import save_pkl_data, load_pkl_data
debug = False

def pre_process(data, image_min_size, image_wh_ratio, text_min_support,
                target_permission_groups, vocab2id, label2id):
    """Pre-processing the <image, texts, permissions> triple.

    The detailed behaviors are list as follow:

    1) image pre-processing.
    1.1) remove data that do not have image or the ratio of width
    and height is strange.

    2) permissions pre-processing.
    2.1) filter out non sensitive permissions.
    2.2) transform permissions to permission groups.

    3) text pre-processing.
    3.1) tokenize and stem layout texts and resource texts.
    3.2) refine embedded texts with english vocabulary and app's
    vocabulary (layout texts and resource texts).

    :param data:
        List, raw data set, [image, [layout_texts, embedded_texts,
        resource_texts], raw permissions] triples.
    :param image_min_size:
        Int, image's width and height should be higher than this.
    :param image_wh_ratio:
        Int or float, the ratio between width and height should not
        be higher than this value.
    :param text_min_support:
        Int, text appeared less than the threshold will be removed
        and present as `UNK` token.
    :param target_permission_groups:
        Dict, map permissions to groups, the key is group name and
        the value is the contained permissions.
    :param vocab2id:
        Dict, transform token to index. If the parameter is None,
        the method will generate the dict based on current texts.
        Note that, 2 special token is added to the dict, that is,
        'UNK' means the out of vocabulary tokens is indexed with
        1, and 'PAD' means the padding of texts is indexed with 0.
        Otherwise, the method will use the given dict and return
        the same dict.
    :param label2id:
        Dict, transform label (i.e., group based permissions) to
        index. It is generated based on the conf.target_groups.

    :return:
        data: List, [image, tokens, permissions] triples;
        vocab2id: Dict, transform token to index;
        label2id: Dict, transform label to index.
    """
    # recorder
    raw_tokens = Counter()  # used to refine the OCR results
    removed_by_permission, removed_by_image = [], []  # record removed data
    # group permissions
    perm2group = pre_process_get_perm2group(target_permission_groups)

    # the first loop
    data_new = []
    for i in range(len(data)):
        img_data, texts, perms = data[i]

        # handle permissions
        sensitive_perms = pre_process_permissions(perms, perm2group)
        if len(sensitive_perms) == 0:
            removed_by_permission.append(data[i])
            continue

        # handle image
        normalized_img_data = pre_process_image(img_data, image_min_size, image_wh_ratio)
        if normalized_img_data is None:
            removed_by_image.append(data[i])
            continue

        # handle texts
        layout_texts, embedded_texts, res_texts = texts
        layout_tokens, stemmed_layout_tokens = pre_process_texts(layout_texts)
        res_tokens, stemmed_res_tokens = pre_process_texts(res_texts)
        raw_tokens.update(layout_tokens + res_tokens)
        # keep the original embedded texts and refine them in the next loop
        texts = [stemmed_layout_tokens, embedded_texts, stemmed_res_tokens]

        # new data
        data_new.append([img_data, texts, perms])

    # the second loop, refine embedded texts
    # spell checking
    speller = pre_process_get_spell_corrector(set(raw_tokens.keys()))
    # record the removed embedded texts
    removed_embedded_texts = Counter()
    for i in range(len(data_new)):
        texts_d = data_new[i][1]
        embedded_texts = texts_d[1]
        stemmed_embedded_tokens, removed_tokens = pre_process_refine_embedded_texts(
            embedded_texts, speller
        )
        removed_embedded_texts.update(removed_tokens)
        # update the new data
        tokens = texts_d[0] + stemmed_embedded_tokens + texts_d[2]
        data_new[i][1] = tokens
    # print statistics information
    pre_process_statistics(data_new, {
        'removed by permissions': removed_by_permission,
        'removed by image': removed_by_image,
        'removed embedded texts': removed_embedded_texts
    })

    # indexing
    if vocab2id is None:
        vocab2id = pre_process_get_vocab_dict(data_new, text_min_support)
    if label2id is None:
        label2id = pre_process_get_perm_dict(target_permission_groups.keys())
    data_indexed = pre_process_indexing(data_new, vocab2id, label2id)
    if debug == True:
        print (data_indexed)
    return data_indexed, vocab2id, label2id


def pre_process_get_perm2group(target_groups):
    to_group = {}
    for k, vs in target_groups.items():
        for v in vs:
            to_group[v] = k
    return to_group


def pre_process_permissions(permissions, perm2group):
    # shorten permission names
    perms = pre_process_shorten_perm_names(permissions)
    # keep sensitive permissions
    perms = [p for p in perms if p in perm2group]
    # map permission to category
    perms = set([perm2group[p] for p in perms])

    return perms


def pre_process_shorten_perm_names(permissions):
    return set([p.split('.')[-1] for p in permissions])


def pre_process_image(img_data, min_size, wh_ratio):
    # resize image
    if img_data is None:
        return None

    # remove images with strange size
    # too small or ratio between width and height is too big or small
    w, h = img_data[1]
    if (w < min_size or h < min_size) or (
            w / h > wh_ratio or h / w > wh_ratio):
        return None

    return img_data


def pre_process_texts(texts):
    # tokenize and stem
    tokens = remove_short_tokens(tokenize_texts(texts))
    stemmed_tokens = stem_tokens(tokens)
    return tokens, stemmed_tokens


def pre_process_get_spell_corrector(app_tokens):
    # update the auto corrector with the App's vocab
    speller = autocorrect.Speller()
    for word in app_tokens:
        speller.nlp_data[word] = 1
    return speller


def pre_process_refine_embedded_texts(texts, speller, is_print=False):
    corrected_sentence = speller.autocorrect_sentence(' '.join(texts).lower())
    corrected_tokens = tokenize_texts([corrected_sentence])
    corrected_tokens, removed_short_tokens = remove_short_tokens(
        corrected_tokens, threshold=2, is_return_removed=True
    )
    stemmed_tokens = stem_tokens(corrected_tokens)

    if is_print:
        print('{} -> {} -> {} -> {}'.format(texts, corrected_sentence,
                                            corrected_tokens, stemmed_tokens))

    return stemmed_tokens, removed_short_tokens


def pre_process_statistics(data, removed_data):
    #print('example:', [data[0][0][:2]] + data[0][1:])  # image meta + other info -> [('LA', (92, 122)), [[], ['4'], ['b41']], {'android.permission.VIBRATE', 'android.permission.WAKE_LOCK'}]
    for removed_name, removed_list in removed_data.items():
        print(removed_name + ':', len(removed_list))
    token_counter = Counter()
    for _, tokens, _ in data:
        token_counter.update(tokens)
    print('vocab:', token_counter)


def pre_process_get_vocab_dict(data, min_support):
    token_counter = Counter()
    for img_data, tokens, permissions in data:
        print(tokens)
        token_counter.update(tokens)

    vocab2id = {'PAD': 0, 'UNK': 1}
    for token, count in token_counter.items():
        if count > min_support:
            vocab2id[token] = len(vocab2id)

    return vocab2id


def pre_process_get_perm_dict(target_permissions):
    return {
        p: i for i, p in enumerate(target_permissions)
    }


def pre_process_indexing(data, vocab2id, label2id):
    data_indexed = []
    token_unk_index = vocab2id['UNK']
    for img_data, tokens, permissions in data:
        tokens_indexed = [vocab2id.get(t, token_unk_index) for t in tokens]
        perms_indexed = [label2id[p] for p in permissions if p in label2id]
        data_indexed.append([img_data, tokens_indexed, perms_indexed])

    return data_indexed


def tokenize_texts(texts):
    if not hasattr(tokenize_texts, 'en_stopwords'):
        tokenize_texts.en_stopwords = set(nltk.corpus.stopwords.words("english"))
    english_stopwords = tokenize_texts.en_stopwords

    result = []
    for text in texts:
        alpha_text = replace_none_alpha(text)
        tokens = nltk.tokenize.word_tokenize(alpha_text)
        tokens = [t.lower() for t in tokens if t not in english_stopwords]
        result.extend(tokens)
    return result


def replace_none_alpha(text):
    result = [
        ch if ch.isalpha() else ' ' for ch in text
    ]
    return ''.join(result)


def remove_short_tokens(tokens, threshold=1, is_return_removed=False):
    if not is_return_removed:
        return [t for t in tokens if len(t) > threshold]
    else:
        removed, result = [], []
        for t in tokens:
            if len(t) > threshold:
                result.append(t)
            else:
                removed.append(t)
        return result, removed


def stem_tokens(tokens):
    if not hasattr(stem_tokens, 'stemmer'):
        stem_tokens.stemmer = nltk.stem.porter.PorterStemmer()
    porter_stemmer = stem_tokens.stemmer
    return [porter_stemmer.stem(t) for t in tokens]


def pre_process_save_results(path_out, data, v2id, l2id):
    if debug == True:
        print("v2id")
        print(v2id)
        print("l2id")
        print(l2id)
        print(len(data),len(v2id),len(l2id))
    save_pkl_data(path_out, [(v2id, l2id), data])


def execute_with_conf(conf, vocab2id=None, label2id=None):
    """
    :param conf:
        PreProcessConf, configuration of path and other opinions.
    :param vocab2id:
        Dict, transform token to index. If the parameter is None,
        the method will generate the dict based on current texts.
        Note that, 2 special token is added to the dict, that is,
        'UNK' means the out of vocabulary tokens is indexed with
        1, and 'PAD' means the padding of texts is indexed with 0.
        Otherwise, the method will use the given dict and return
        the same dict.
    :param label2id:
        Dict, transform label (i.e., group based permissions) to
        index. It is generated based on the conf.target_groups.

    :return:
        result: Tuple, (data, vocab2id, label2id).
    """
    # load data
    data = load_pkl_data(conf.path_data_in)
    if debug == True:
        print("length of loaded data" + str(len(data)))
        #print(data[0])
    
    # pre-process
    result = pre_process(
        data, conf.image_min_size, conf.image_wh_ratio, conf.text_min_support,
        conf.target_groups, vocab2id, label2id
    )
    # save results
    pre_process_save_results(conf.path_data_out, *result)
    if debug == True:
        print(result)
    return result



def total_example():
    from conf import PreProcessConf, target_groups

    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data')

    conf_benign = PreProcessConf(
        # path
        path_data_in=os.path.join(path_data, 'example', 'raw_benign_debug.pkl'),
        path_data_out=os.path.join(path_data, 'example', 'processed_benign_debug.pkl'),
        # image
        image_min_size=5,
        image_wh_ratio=10,
        # text
        text_min_support=5,
        # permissions
        target_groups=target_groups,
    )
    '''
    conf_mal = PreProcessConf(
        # path
        path_data_in=os.path.join(path_data, 'example', 'raw_data.mal.pkl'),
        path_data_out=os.path.join(path_data, 'example', 'data.mal.pkl'),
        # image
        image_min_size=5,
        image_wh_ratio=10,
        # text
        text_min_support=5,
        # permissions
        target_groups=target_groups,
    )
    '''
    print('benign')
    _, v2id, l2id = execute_with_conf(conf_benign)
    '''
    print('malicious')
    execute_with_conf(conf_mal, v2id, l2id)
    '''
def new_apk(newApkOutputPath):
    from conf import PreProcessConf, target_groups

    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data')

    conf_apk = PreProcessConf(
        # path
        path_data_in=os.path.join(path_data, newApkOutputPath, 'raw_extraction_data.pkl'),
        path_data_out=os.path.join(path_data, newApkOutputPath, 'processed_extraction_data.pkl'),
        # image
        image_min_size=5,
        image_wh_ratio=10,
        # text
        text_min_support=5,
        # permissions
        target_groups=target_groups,
    )
    print('new apk')
    execute_with_conf(conf_apk)


def main():
    import sys
    args = sys.argv[1:]
    if '--outlier_detection' in args:
        newApkOutputPath = sys.argv[1]
        new_apk(newApkOutputPath)
    else:
        total_example()


if __name__ == '__main__':
    main()
