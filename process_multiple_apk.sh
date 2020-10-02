#!/bin/sh

DI=/mnt/c/intern/deepintent/
APK=/mnt/c/intern/deepintent/data/example/benign/
FINALDIR=/mnt/c/intern/deepintent/data/newApk
cd $APK
for app in `ls *.apk`; do
    mkdir $FINALDIR/$app
    NEWAPK=$FINALDIR/$app #single dir for each APK under /data/newApk
    mkdir $NEWAPK/decoded
    NEWAPKDECODED=$NEWAPK/decoded #decoded inside the single dir
    mkdir $NEWAPK/apk
    cp -r $APK/$app $NEWAPK/apk #copy apk file into a sub-dir to seperate raw apk from rest of outputs
    cd $DI
    bash new_apkLocal.sh $NEWAPK $NEWAPKDECODED $app $FINALDIR/
done
#${string%substring}
