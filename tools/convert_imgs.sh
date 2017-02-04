#! /usr/bin/env sh
# author:liuke
# date:01/03/2017
# resize image.
# usage: sh convert_imgs.sh
imagedata_path=/home/ke/utils/img/small_data/
for i in ${imagedata_path}*; do
    #c=`basename $i`
    #label=$(($label+1))
    for j in `ls $i/*.jpg`; do
        #echo "$j ${label}"
	echo "processing $j"
        convert $j -resize 256x256! $j
    done
done
#echo -e "\n"
