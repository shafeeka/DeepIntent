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

debug = False

def predict(predict_conf):
    # load data
    result = load_pkl_data(predict_conf.path_data)
    _, data, wid_name_list = result

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
    x_test, y_test = data_test
    y_predict = model.predict(x_test)
    y_true = y_test.tolist()
    scores = evaluate(y_true, y_predict, predict_conf.threshold)
    label_names = [meta_id2label[i] for i in range(len(meta_id2label))]
    if debug:
        print(len(x_test))
        print(len(y_true))
        print(len(y_test))
    


    if '--outlier_detection' in sys.argv[1:]:
        #print prediction results
        predictionPath = sys.argv[1] 
        f = open(predictionPath, "a")
        f.write(sys.argv[4] + "\n") #apppend app name first
         #append widget name

        for i in range(len(y_true)):
            f.write(wid_name_list[i] + "\n")
            f.write("y_true is: " + str(y_true[i]) + "\n" + "y_predict is: " + str(y_predict[i]) + "\n\n")
        
        f.close()

        # print metric results
        metricsPath = sys.argv[2]
        f2 = open(metricsPath, "a")
        f2.write(sys.argv[4] + "\n") #apppend app name first
        
        scoreType = ["precision", "recall", "f1", "acc","support"]
        thresholdScores = scores[1]

        for k in range(len(meta_id2label)): #8 - print permission group
            f2.write(meta_id2label[k] + "\n")
            for s in range(len(thresholdScores)): #5 - print the 5 scores for the permission group
                f2.write(scoreType[s] + " = " + str(thresholdScores[s][k]) + "\t")
            f2.write("\n")
        f2.close()

    else:
        # save predictions
        save_pkl_data(predict_conf.path_predictions, [y_predict, y_test])
        display_scores(scores, label_names)

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

def new_apk(newApkInputPath):
    import os

    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data')
    predict_conf = PredictConf(
        # path
        path_data=os.path.join(path_data, newApkInputPath, 'processed_extraction_data.pkl'),
        path_meta=os.path.join(path_data, 'total', 'deepintent.meta'),
        path_model=os.path.join(path_data, 'total', 'deepintent.model'),
        path_predictions=os.path.join(path_data, newApkInputPath, 'new.apk.predictions'),
        # prediction
        threshold=0.5
    )

    predict(predict_conf)


def main():
    args = sys.argv[1:]
    if '--outlier_detection' in args:
        single_apk = True
        newApkInputPath = sys.argv[3]
        new_apk(newApkInputPath)
    else:
        total_example()


if __name__ == '__main__':
    main()
