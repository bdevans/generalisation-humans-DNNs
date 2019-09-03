#!/bin/bash
img_path="/storage/data/imagenet_2012"
list_dir="image_names"
out_dir="images"

for file in $(ls $list_dir/)
do
  category=$(echo $file | cut -d "." -f 1)
  echo $category
  mkdir $category

  while IFS= read -r image
  do
    folder=$(echo $image | cut -d "_" -f 1)
    #echo "Copying $img_path/$folder/$image into $category"
    cp $img_path/$folder/$image $out_dir/$category/
  done < "$list_dir/$file"
done
