import os
import sys

path_apk = sys.argv[1]
#'/mnt/c/intern/deepintent/data/example/benignAll/'
path_decode = sys.argv[2]
#'/mnt/c/intern/deepintent/data/example/benign_decoded/'
path_gator_APK = sys.argv[3]

if not os.path.exists(path_decode):
    os.mkdir(path_decode)

for filename in os.listdir(path_apk):
    apk_name, ext_name = os.path.splitext(filename)
    print(apk_name)

    path_in = os.path.join(path_apk, filename)
    path_out = os.path.join(path_decode, apk_name)
    os.system('java -jar ' + path_gator_APK + '/apktool.jar d {} -o {}'.format(path_in, path_out))
#java -jar /home/shafeeka/deepintent/IconWidgetAnalysis/Static_Analysis/gator-IconIntent/AndroidBench/apktool.jar