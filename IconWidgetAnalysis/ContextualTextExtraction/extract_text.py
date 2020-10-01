# -*- coding: UTF-8 -*-

import os

from conf import check_conf, ExtractionConf
from tools import save_pkl_data, ProcessPrinter
from load_data import load_data
from handle_layout_text import handle_layout_text
from handle_embedded_text import extract_drawable_image, load_east_model, extract_embedded_text
from handle_resource_text import handle_resource_text

debug = False
def extract_contextual_texts(data, drawable_images, path_app, path_east,
                             search_range='parent',
                             ocr_size=(320, 320), ocr_padding=0.1, enable_ocr_cache=True,
                             is_translate=True, enable_translate_cache=True,
                             log_level=0):
    # extract layout texts
    print('extracting layout texts')
    layout_texts = extract_layout_texts(data, path_app, search_range, log_level)

    # extract embedded texts
    print('extracting embedded texts')
    app2lang = get_app2lang(data, layout_texts)
    east_model = load_east_model(path_east)
    embedded_texts = extract_embedded_texts(data, drawable_images, app2lang, east_model,
                                            ocr_size, ocr_padding, enable_ocr_cache, log_level)

    # translate layout texts and embedded texts
    if is_translate:
        print('translating')
        layout_texts = translate_texts(data, layout_texts, enable_translate_cache, log_level)
        embedded_texts = translate_texts(data, embedded_texts, enable_translate_cache, log_level)

    # extract resource texts
    print('extracting resource texts')
    resource_texts = extract_resource_texts(data, log_level)

    # merge extracted texts
    assert len(data) == len(layout_texts) == len(embedded_texts) == len(resource_texts)
    results = [[layout_texts[i], embedded_texts[i], resource_texts[i]] for i in range(len(data))]
    return results


def extract_drawable_images(data_pa, path_app, log_level=0):
    """Extract drawable images for each (app, img, layout) tuple.

    :param data_pa:
        List, each row contains (app, img, layout, permissions) tuple.
    :param path_app:
        String, the path of the decoded Apps.
    :param log_level:
        Int, 0 to 2, silent, or normal (process bar), or verbose mode.

    :return:
        List, compressed images.
    """
    results = []
    log_helper = ProcessPrinter(len(data_pa) / 20, log_level)
    for app_name, img_name, layout, _ in data_pa:
        result, result_path, result_traces = extract_drawable_image(app_name, img_name, path_app)
        results.append(result)
        log_helper.update('[image]', app_name, img_name, layout, ':',
                          (result[0], result[1]) if result is not None else result)
    log_helper.finish()

    return results


def extract_layout_texts(data_pa, path_app, search_range, log_level=0):
    results = []
    log_helper = ProcessPrinter(len(data_pa) / 20, log_level)
    for app_name, img_name, layout, _ in data_pa:
        result = handle_layout_text(app_name, img_name, layout, path_app, search_range)
        results.append(result)
        log_helper.update('[layout]', app_name, img_name, layout, ':', result)
    log_helper.finish()

    return results


def get_app2lang(data_pa, layout_texts):
    from translate_text import check_default_language

    # collect all the layout texts appeared in the app
    app_texts = {}  # app -> all the layout texts
    for i in range(len(data_pa)):
        app_name = data_pa[i][0]
        if app_name not in app_texts:
            app_texts[app_name] = []
        app_texts[app_name].extend(layout_texts[i])

    app2lang = {app_name: check_default_language(texts) for app_name, texts in app_texts.items()}
    return app2lang


def extract_embedded_texts(data_pa, drawable_images, app2lang, east_model,
                           ocr_size, ocr_padding, enable_cache=True, log_level=0):
    results = []
    log_helper = ProcessPrinter(len(data_pa) / 20, log_level)
    for i in range(len(data_pa)):
        app_name, img_name, layout, _ = data_pa[i]
        result = extract_embedded_text(app_name, img_name, drawable_images[i], east_model,
                                       app2lang[app_name], 'english',
                                       ocr_size, ocr_padding, enable_cache)
        results.append(result)
        log_helper.update('[embedded]', app_name, img_name, layout, ':', result)
    log_helper.finish()

    return results


def translate_texts(data_pa, texts, enable_cache=True, log_level=0):
    from translate_text import translate_any_to_english
    assert len(data_pa) == len(texts)

    results = []
    log_helper = ProcessPrinter(sum([len(t) for t in texts]) / 20, log_level)
    for i in range(len(data_pa)):
        app_name, img_name, layout, _ = data_pa[i]
        translated = []
        for t in texts[i]:
            r = translate_any_to_english(t, enable_cache)
            translated.append(r)
            log_helper.update('[translate]', app_name, img_name, layout, ':', t, '->', r)
        results.append(translated)
    log_helper.finish()

    return results


def extract_resource_texts(data_pa, log_level=0):
    results = []
    log_helper = ProcessPrinter(len(data_pa) / 20, log_level)
    for app_name, img_name, layout, _ in data_pa:
        result = handle_resource_text(img_name)
        results.append(result)
        log_helper.update('[res]', app_name, img_name, layout, ':', result)
    log_helper.finish()

    return results


def execute_with_conf(conf):
    # load program analysis results, format: app, image, layout, permissions
    print('loading program analysis results')
    data_pa = load_data(conf.path_pa)
    if debug==True:
        print(data_pa[0:2])
    # extract drawable images
    print('extracting drawable images')
    log_level = check_conf(conf.log_level, {0, 1, 2}, 0) #log_level = 2
    drawable_images = extract_drawable_images(data_pa, conf.path_app, conf.log_level)

    # extract texts, format: layout_texts, embedded_texts, resource_texts
    print('extracting texts')
    search_range = check_conf(conf.layout_text_range, {'parent', 'total'}, 'parent')
    enable_ocr_cache = check_conf(conf.enable_ocr_cache, {True, False}, True)
    is_translate = check_conf(conf.enable_translate, {True, False}, True)
    enable_translate_cache = check_conf(conf.enable_translate_cache, {True, False}, True)
    texts = extract_contextual_texts(data_pa, drawable_images, conf.path_app, conf.path_east,
                                     search_range,
                                     (conf.ocr_width, conf.ocr_height), conf.ocr_padding, enable_ocr_cache,
                                     is_translate, enable_translate_cache,
                                     log_level)

    # merge and save the triple, <image, texts, permissions>
    print('finished and save')
    assert len(data_pa) == len(drawable_images) == len(texts)
    # format: [(compressed_img), [[layout_texts], [embedded_texts], [res_texts]], {permissions}]
    data = [[drawable_images[i]] + [texts[i]] + [data_pa[i][-1]] for i in range(len(data_pa))]
    if debug==True:
        print([[drawable_images[1]] + [texts[1]] + [data_pa[1]][-1]])
    save_pkl_data(conf.path_save, data)



def example():
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', '..', 'data')
    example_conf = ExtractionConf(
        # path
        path_pa=os.path.join(path_data, 'text_example', 'total', 'example.zip'),
        path_app=os.path.join(path_data, 'text_example', 'total', 'apk_decoded'),
        path_east=os.path.join(path_data, 'frozen_east_text_detection.pb'),
        path_save=os.path.join(path_data, 'text_example', 'total', 'data.pkl'),
        # log
        log_level=2,
        # layout text extraction
        layout_text_range='parent',
        # embedded text extraction
        ocr_width=320,
        ocr_height=320,
        ocr_padding=0.05,
        enable_ocr_cache=True,
        # translation
        enable_translate=True,
        enable_translate_cache=True
    )
    print(example_conf)
    execute_with_conf(example_conf)

def new_apk():
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', '..', 'data')
    apk_conf = ExtractionConf(
        # path
        path_pa=os.path.join(path_data, 'new_apk', 'new_apk_output', 'outputP.csv'),
        path_app=os.path.join(path_data, 'new_apk', 'new_apk_decoded'),
        path_east=os.path.join(path_data, 'frozen_east_text_detection.pb'),
        path_save=os.path.join(path_data, 'new_apk', 'raw_extraction_data.pkl'),
        # log
        log_level=2,
        # layout text extraction
        layout_text_range='parent',
        # embedded text extraction
        ocr_width=320,
        ocr_height=320,
        ocr_padding=0.05,
        enable_ocr_cache=True,
        # translation
        enable_translate=True,
        enable_translate_cache=True
    )
    print("extracting from new apk")
    execute_with_conf(apk_conf)

def total_example():
    if debug==True:
        print("testing pass")
    # path
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', '..', 'data')
    # conf
    benign_conf = ExtractionConf(
        # path
        path_pa=os.path.join(path_data, 'example', 'outputP.csv'),
        path_app=os.path.join(path_data, 'example', 'benign_decoded'),
        path_east=os.path.join(path_data, 'frozen_east_text_detection.pb'),
        path_save=os.path.join(path_data, 'example', 'raw_benign_debug.pkl'),
        # log
        log_level=2,
        # layout text extraction
        layout_text_range='parent',
        # embedded text extraction
        ocr_width=320,
        ocr_height=320,
        ocr_padding=0.05,
        enable_ocr_cache=True,
        # translation
        enable_translate=True,
        enable_translate_cache=True
    )
    '''
    malicious_conf = ExtractionConf(
        # path
        path_pa=os.path.join(path_data, 'example', 'malicious_pa.zip'),
        path_app=os.path.join(path_data, 'example', 'malicious_decoded'),
        path_east=os.path.join(path_data, 'frozen_east_text_detection.pb'),
        path_save=os.path.join(path_data, 'example', 'raw_data.mal.pkl'),
        # log
        log_level=1,
        # layout text extraction
        layout_text_range='parent',
        # embedded text extraction
        ocr_width=320,
        ocr_height=320,
        ocr_padding=0.05,
        enable_ocr_cache=True,
        # translation
        enable_translate=True,
        enable_translate_cache=True
    )
    '''
    print('benign')
    execute_with_conf(benign_conf)
    '''
    print('malicious')
    execute_with_conf(malicious_conf)
    '''


def main():
    """This script can be directly used in command line.

    If the arguments contain `--example`, the script will run a simple example
    to extract contextual texts stored in `data/text_example/total` folder with
    prepared program analysis outputs (example.zip). *Note that, please decode
    the APKs into `data/text_example/total/apk_decoded` to run the example.*

    If the arguments contain `--total_example`, the script will handle benign and
    malicious APKs stored in `data/example` (should be downloaded from BaiduYun).
    *Note that, 1) please run program analysis and put the zipped output in the
    data folder (e.g., `benign_pa.csv` contained in `benign_pa.zip`), 2) decode
    the APKs (e.g., `benign_decoded` for benign APKs).*

    Otherwise, users could specify data paths through arguments, 4 arguments are
    necessary, namely `--path_pa` indicates the path of program analysis results,
    `--path_app` indicates the decoded apps, `--path_east` means the pre-trained
    EAST model (frozen_east_text_detection.pb, can be download from BaiduYun), and
    `--path_save` indicates where to save the outputs. Other optional arguments
    please see the README.md file.
    """
    import sys
    #from pycallgraph import PyCallGraph
    #from pycallgraph.output import GraphvizOutput
    #graphviz = GraphvizOutput()
    #graphviz.output_file = 'basic.png'
    args = sys.argv[1:]
    # ./extract.py --ex fafdsa   --example <real file>  --total_example  -datafile <datafile> 
    # ./extract.py <datafile> --example --total_example
    if '--outlier_detection' in args:
        #for i in range(len(args)):
        #    if '-datafile' ==  args[i] and (i+1 < len(args)):
        #        datafile = args[i+1] 
        #with PyCallGraph(output=graphviz):
        new_apk()

    # example or total example
    # elif '--example' in args:
    #     example()
    elif '--total_example' in args:
        total_example()
    # else:
    #     # extraction texts based on arguments
    #     from conf import ExtractionConfArgumentParser
    #     parser = ExtractionConfArgumentParser()
    #     args_conf = parser.parse(args)
    #     execute_with_conf(args_conf)


if __name__ == '__main__':
    main()
