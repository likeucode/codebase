find training_task12_images/ -name *.jpg>./train_test_list/train.txt
find training_task1_gt_xml/ -name *.xml>./train_test_list/train_label.txt

find task12_images/ -name *.jpg>./train_test_list/test_tmp.txt
find task1_gt_xml/ -name *.xml>./train_test_list/test_label.txt

sort train.txt >train_sorted.txt
sort train_label.txt >train_label_sorted.txt
sort test_tmp.txt >test_sorted.txt
sort test_label.txt >test_label_sorted.txt


paste -d' ' train_sorted.txt train_label_sorted.txt >> trainval.txt
paste -d' ' test_sorted.txt test_label_sorted.txt >> test.txt

./build/tools/get_image_size /home/ke/ocr/TextBoxes/data/icdar/icdar2013/ /home/ke/ocr/TextBoxes/data/icdar/icdar2013/train_test_list/trainval.txt /home/ke/ocr/TextBoxes/data/icdar/icdar2013/train_test_list/train_name_size.txt


./build/tools/get_image_size /home/ke/ocr/TextBoxes/data/icdar/icdar2013/ /home/ke/ocr/TextBoxes/data/icdar/icdar2013/train_test_list/test_final.txt /home/ke/ocr/TextBoxes/data/icdar/icdar2013/train_test_list/test_name_size.txt


cat trainval.txt | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > trainval_random.txt
mv trainval_random.txt trainval.txt