#! /bin/sh
#windows
IC3PATH=/mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/ic3
cd $IC3PATH
appDir=$1 
forceAndroidJar=/mnt/c/Users/ASUS/AppData/Local/Android/Sdk/platforms/android-18/android.jar
rm -rf testspace
mkdir testspace
rm -rf ic3output
mkdir ic3output
var=0
for appPath in `ls $appDir/*.apk`; do
    appName=`basename $appPath .apk`
    retargetedPath=$IC3PATH/testspace/$appName.apk

    #mysql -h127.0.0.1 -uroot -e 'drop database if exists cc; create database cc'
    #mysql -h127.0.0.1 -uroot cc < /mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/ic3/schema

    rm -rf $IC3PATH/ic3output/$appName
    mkdir $IC3PATH/ic3output/$appName

    #echo java -Xmx24000m -jar /mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/ic3/RetargetedApp.jar $forceAndroidJar $appPath $retargetedPath
    #echo java -Xmx24000m -jar /mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/ic3/ic3-0.2.0-full.jar -apkormanifest $appPath -input $retargetedPath -cp $forceAndroidJar -db cc.properties -dbname cc1 -protobuf /mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/output/$appName
    timeout 1800 java -Xmx24000m -jar $IC3PATH/RetargetedApp.jar $forceAndroidJar $appPath $retargetedPath
    timeout 1800 java -Xmx24000m -jar $IC3PATH/ic3-0.2.0-full.jar -apkormanifest $appPath -input $retargetedPath -cp $forceAndroidJar -db cc.properties -dbname cc -protobuf $IC3PATH/ic3output/$appName
    var=$((var+1))
done
echo "finished $var apk files."
#    gtimeout 1800 java -Xmx24000m

# #ubuntu
# IC3PATH=/home/shafeeka/deepintent/IconWidgetAnalysis/Static_Analysis/ic3
# cd $IC3PATH
# appDir=$1 #/mnt/c/intern/deepintent/data/example/benign
# forceAndroidJar=/home/shafeeka/Android/Sdk/platforms/android-18/android.jar
# rm -rf testspace
# mkdir testspace
# rm -rf ic3output
# mkdir ic3output
# var=0
# for appPath in `ls $appDir/*.apk`; do
#     appName=`basename $appPath .apk`
#     retargetedPath=$IC3PATH/testspace/$appName.apk

#     #mysql -ushafeeka -pjiaozhuys05311 -e 'drop database if exists cc; create database cc'
#     #mysql -ushafeeka -pjiaozhuys05311 cc < $IC3PATH/schema

#     rm -rf $IC3PATH/ic3output/$appName
#     mkdir $IC3PATH/ic3output/$appName

#     timeout 1800 java -Xmx24000m -jar $IC3PATH/RetargetedApp.jar $forceAndroidJar $appPath $retargetedPath
#     timeout 1800 java -Xmx24000m -jar $IC3PATH/ic3-0.2.0-full.jar -apkormanifest $appPath -input $retargetedPath -cp $forceAndroidJar -db cc.properties -dbname cc -protobuf $IC3PATH/ic3output/$appName
#     var=$((var+1))
# done
# echo "finished $var apk files."

