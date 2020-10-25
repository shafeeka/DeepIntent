#!/bin/sh

DI=/home/shafeeka/deepintent/
GATORAPKTOOL=/home/shafeeka/deepintent/IconWidgetAnalysis/Static_Analysis/gator-IconIntent/AndroidBench
if [ "$#" -eq "0" ] #to check if argument passed
then
    DECODED=/home/shafeeka/deepintent/data/new_apk/new_apk_decoded/ #dir to store the decoded apk files
    APK=/home/shafeeka/deepintent/data/example/newApk/ #input apk files dir
    PREDRES=/home/shafeeka/deepintent/data/new_apk/predictions.txt
    METRICSRES=/home/shafeeka/deepintent/data/new_apk/metrics.txt
    APPNAME="None"
else 
    APK=$1
    DECODED=$2
    APPNAME=$3
    PREDRES=$4/predictions.txt #created all txt files prior to running
    METRICSRES=$4/metrics.txt
    ERRORRES=$4/error_processing.txt
fi
#run static analysis : edit the path variables script prior
cd $DI/IconWidgetAnalysis/Static_Analysis/
bash clearAllServer.sh
bash runImg2widgets.sh $APK/apk &>> $APK/log.txt
cp -r outputP.csv $APK
#check if outputP.csv is empty
if [ "$(wc -l <$APK/outputP.csv)" -eq 1 ]
then
    echo "outputP.csv of $APPNAME is empty" >> $ERRORRES
    exit
fi
#run decode.py
cd $DI/data/text_example/total/
python3 decode.py $APK/apk $DECODED $GATORAPKTOOL &>> $APK/log.txt
#run contextual text extraction
cd $DI/IconWidgetAnalysis/ContextualTextExtraction
python3 extract_text.py $APK $DECODED '--outlier_detection' &>> $APK/log.txt
#run pre_process.py
cd $DI/IconBehaviorLearning
python3 pre_process.py $APK '--outlier_detection' &>> $APK/log.txt
#run predict.py
python3 predict.py $PREDRES $METRICSRES $APK $APPNAME '--outlier_detection' &>> $APK/log.txt
if [ "$?" -ne "0" ]
then
    echo "predict.py of $APPNAME is unsuccessful" >> $ERRORRES
fi