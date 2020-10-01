#!/bin/sh

DI=/home/shafeeka/deepintent
APK=/home/shafeeka/deepintent/data/example/newApk/ #input apk files dir
DECODED=/home/shafeeka/deepintent/data/new_apk/new_apk_decoded/ #dir to store the decoded apk files
#run static analysis : edit the path variables script prior
cd $DI/IconWidgetAnalysis/Static_Analysis/
sh runImg2widgets.sh 
cp -r outputP.csv $DI/data/new_apk/new_apk_output/
#run decode.py
cd $DI/data/text_example/total/
python3 decode.py $APK $DECODED
#run contextual text extraction
cd $DI/IconWidgetAnalysis/ContextualTextExtraction
python3 extract_text.py '--outlier_detection'
#run pre_process.py
cd $DI/IconBehaviorLearning
python3 pre_process.py '--outlier_detection'
#run predict.py
python3 predict.py /home/shafeeka/deepintent/data/new_apk/predictions.txt /home/shafeeka/deepintent/data/new_apk/metrics.txt '--outlier_detection'