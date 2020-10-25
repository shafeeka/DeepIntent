#!/bin/sh

DI=/home/shafeeka/deepintent/
APK=/home/shafeeka/deepintent/data/example/benign/
FINALDIR=/home/shafeeka/deepintent/data/newApk
cd $APK
for app in `ls *.apk`; do
    mkdir $FINALDIR/$app
    NEWAPK=$FINALDIR/$app #single dir for each APK under /data/newApk
    mkdir $NEWAPK/decoded
    NEWAPKDECODED=$NEWAPK/decoded #decoded inside the single dir
    mkdir $NEWAPK/apk
    cp -r $APK/$app $NEWAPK/apk #copy apk file into a sub-dir to seperate raw apk from rest of outputs
    cd $DI
    bash new_apk.sh $NEWAPK $NEWAPKDECODED $app $FINALDIR
done