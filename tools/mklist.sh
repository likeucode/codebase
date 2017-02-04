#! /usr/bin/env sh
# author:liuke
# date:12/28/2016
# It produces list file of image datase.
# usage: sh mklist.sh
label=-1
imagedata_path=/home/ke/face_data/val/
for i in ${imagedata_path}*; do
    # c=`basename $i`
    label=$(($label+1))
    for j in `ls $i/*.jpg`; do
        echo "$j ${label}"
    done
done
#echo -e "\n"
