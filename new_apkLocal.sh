#!/bin/sh

DI=/mnt/c/intern/deepintent
GATORAPKTOOL=/mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/gator-IconIntent/AndroidBench
if [ "$#" -eq "0" ] #to check if argument passed
then
    DECODED=/mnt/c/intern/deepintent/data/new_apk/new_apk_decoded/ #dir to store the decoded apk files
    APK=/mnt/c/intern/deepintent/data/example/newApk/ #input apk files dir
    PREDRES=/mnt/c/intern/deepintent/data/new_apk/predictions.txt
    METRICSRES=/mnt/c/intern/deepintent/data/new_apk/metrics.txt
    APPNAME="None"
else 
    APK=$1
    DECODED=$2
    APPNAME=$3
    PREDRES=$4/predictions.txt #created txt files prior to running
    METRICSRES=$4/metrics.txt
fi
#run static analysis : edit the path variables script prior
cd $DI/IconWidgetAnalysis/Static_Analysis/
bash clearAll.sh
bash runImg2widgetsLocal.sh $APK/apk &>> $APK/log.txt
cp -r outputP.csv $APK
#check if outputP.csv us empty
if [ "$(wc -l <$APK/outputP.csv)" -eq 1 ]
then
    echo "outputP.csv of $APPNAME is empty. Exiting..."
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
    echo "error in predict.py"
fi