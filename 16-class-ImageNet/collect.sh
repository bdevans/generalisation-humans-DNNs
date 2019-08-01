#!/bin/bash
img_path="/storage/data/imagenet_2012"
list_path="image_names"

for file in $(ls $list_path/)
do
  category=$(echo $file | cut -d "." -f 1)
  echo $category
  mkdir $category

  while IFS= read -r image
  do
    folder=$(echo $image | cut -d "_" -f 1)
    #echo "Copying $img_path/$folder/$image into $category"
    cp $img_path/$folder/$image $category/
  done < "$list_path/$file"
done
