#!/bin/sh

DISA=/home/shafeeka/deepintent/IconWidgetAnalysis/Static_Analysis
#clear result.txt and selectedAPK.txt
rm result.txt
rm selectedAPK.txt
touch result.txt
touch selectedAPK.txt
#clear output, permission_output, dot_output, img2widgets dirs
rm -rf output
mkdir output
rm -rf permission_output
mkdir permission_output
rm -rf dot_output
mkdir dot_output
rm -rf img2widgets
mkdir img2widgets
#in gator-IconIntent: clear output and dot_output
cd $DISA/gator-IconIntent
rm -rf output
mkdir output
rm -rf dot_output
mkdir dot_output
#in ImagetoWidgetAnalyzer: remove selectedAPK.txt
cd $DISA/ImageToWidgetAnalyzer
rm selectedAPK.txt
# clear WidImageResolver/output
cd $DISA/WidImageResolver
rm -rf output
mkdir output
