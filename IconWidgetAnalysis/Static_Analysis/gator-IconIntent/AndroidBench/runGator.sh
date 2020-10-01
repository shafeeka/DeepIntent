#! /bin/sh
export ADK=/mnt/c/Users/ASUS/AppData/Local/Android/Sdk
export GatorRoot=/mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/gator-IconIntent
var=1
for app in `ls /mnt/c/intern/deepintent/data/example/benign/*.apk`; do
    echo "analyzing $app"
    python3 /mnt/c/intern/deepintent/IconWidgetAnalysis/Static_Analysis/gator-IconIntent/AndroidBench/runGatorOnApk.py $app -client WTGDemoClient
    echo "done analyzing $app"
    var=$((var+1))
done
echo "analyzed $var files"
