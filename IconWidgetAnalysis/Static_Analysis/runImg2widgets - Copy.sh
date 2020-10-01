#!/bin/bash

DISA=/home/shafeeka/deepintent/IconWidgetAnalysis/Static_Analysis
SDK=/home/shafeeka/Android
APK=/home/shafeeka/deepintent/data/example/newApk
#APK=/home/shafeeka/deepintent/data/example/benign
SQL="jdbc:mysql://127.0.0.1:3306/cc?user=shafeeka&password=jiaozhuys05311&serverTimezone=GMT"
cd $DISA/gator-IconIntent/
python3 gator.py $APK $SDK $DISA/gator-IconIntent $DISA/result.txt
cp -r $DISA/gator-IconIntent/output/ $DISA/
cd $DISA/
#get apk names list
python3 getAPKNames.py $DISA/output $DISA/selectedAPK.txt
#mkdir img2widgets
#mkdir permission_output
#mkdir dot_output 
cp -r selectedAPK.txt $DISA/ImageToWidgetAnalyzer/
#run icon-widget-handler association
cp -r $DISA/gator-IconIntent/output/ $DISA/WidImageResolver
cd $DISA/WidImageResolver/
java -jar $DISA/wid.jar $APK/
cp -r $DISA/WidImageResolver/output/ $DISA
cd $DISA/
java -jar ImageToWidgetAnalyzer.jar $DISA/output $DISA/output $DISA/ $DISA/selectedAPK.txt
#run ic3
sh $DISA/ic3/runic3.sh $APK
#run handler-permission association
for app in `ls $APK/*.apk`; do
    echo $app
    java -jar APKCallGraph.jar $app $APK $DISA/img2widgets $DISA/permission_output/ $DISA/ic3/ic3output $SDK/platforms/android-18/android.jar $DISA/APKCallGraph/SourcesAndSinks.txt $DISA/APKCallGraph/AndroidCallbacks.txt $DISA/dot_output/ $SQL
done
#combine results and get 1-to-more mapping using 1tomore.txt
python3 combine.py $DISA/permission_output/
python3 map1tomore.py $DISA/permissions.csv $DISA/1tomore.txt $DISA/outputP.csv


