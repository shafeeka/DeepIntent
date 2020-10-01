# -*- coding: UTF-8 -*-

"""Use the pre-trained model to predict new data.
"""
import sys
import keras

from conf import PredictConf
from tools import save_pkl_data, load_pkl_data
from layers import CoAttentionParallel
from prepare import prepare_data
from metrics import evaluate, display_scores


def predict(predict_conf):
    # load data
    _, data = load_pkl_data(predict_conf.path_data)

    # load model meta data
    meta = load_pkl_data(predict_conf.path_meta)
    meta_image_shape = meta['ModelConf'].img_shape
    meta_re_sample_type = meta['ModelConf'].img_re_sample
    meta_text_len = meta['ModelConf'].text_length
    meta_label_num = len(meta['label2id'])
    meta_id2label = {v: k for k, v in meta['label2id'].items()}

    # load model
    model = keras.models.load_model(predict_conf.path_model, custom_objects={
        "CoAttentionParallel": CoAttentionParallel
    })

    # prepare data
    _, _, data_test = prepare_data(data, meta_image_shape, meta_re_sample_type,
                                   meta_text_len, meta_label_num, 0, 0)

    # predict with trained model
    print(data_test)
    x_test, y_test = data_test
    y_predict = model.predict(x_test)

    y_true = y_test.tolist()
    txtPath = sys.argv[1]
    print(len(x_test))
    print(len(y_true))
    print(len(y_test))
    f = open(txtPath, "a")
    for i in range(len(y_true)):
        f.write(str(y_true[i]) + "\t" + str(y_predict[i]) + "\n")
        #f.write(str(x_test) + "\t" + str(y_true[i]) + "\t" + str(y_predict[i]) + "\n")
    f.close()
    #save to txt file


    '''
    # save predictions
    save_pkl_data(predict_conf.path_predictions, [y_predict, y_test])
    '''
    # print metric results
    scores = evaluate(y_true, y_predict, predict_conf.threshold)
    label_names = [meta_id2label[i] for i in range(len(meta_id2label))]
    #display_scores(scores, label_names)
    txtPath2 = sys.argv[2]
    f2 = open(txtPath2, "a")
    f2.write("PERMISSION LABELS:")
    scoreType = ["precision", "recall", "f1", "acc"]
    for j in range(len(label_names)):
        f2.write(str(label_names[j]) + "\t")
    for k in range(len(scores)):
        for s in range(len(scoreType)):
            f2.write(scoreType[s] + "=" + str(scores[k][s]) + "\t")
        f2.write("\n")
    f2.close()

def total_example():
    import os

    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data')
    predict_conf = PredictConf(
        # path
        path_data=os.path.join(path_data, 'new_apk', 'processed_extraction_data.pkl'),
        path_meta=os.path.join(path_data, 'total', 'deepintent.meta'),
        path_model=os.path.join(path_data, 'total', 'deepintent.model'),
        path_predictions=os.path.join(path_data, 'new_apk', 'new.apk.predictions'),
        # prediction
        threshold=0.5
    )

    predict(predict_conf)

def new_apk():
    import os

    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data')
    predict_conf = PredictConf(
        # path
        path_data=os.path.join(path_data, 'new_apk', 'processed_extraction_data.pkl'),
        path_meta=os.path.join(path_data, 'total', 'deepintent.meta'),
        path_model=os.path.join(path_data, 'total', 'deepintent.model'),
        path_predictions=os.path.join(path_data, 'new_apk', 'new.apk.predictions'),
        # prediction
        threshold=0.5
    )

    predict(predict_conf)


def main():
    args = sys.argv[1:]
    if '--outlier_detection' in args:
        new_apk()
    else:
        total_example()


if __name__ == '__main__':
    main()
