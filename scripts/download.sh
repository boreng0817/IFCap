#!/bin/bash

links=( "https://github.com/FeiElysia/ViECap/releases/download/checkpoints/evaluation.zip"
    "https://github.com/FeiElysia/ViECap/releases/download/checkpoints/annotations.zip"
    "https://github.com/FeiElysia/ViECap/releases/download/checkpoints/checkpoints.zip" )

for link in ${links[@]};
do
    wget $link;
done

for f in annotations.zip checkpoints.zip evaluation.zip;
do
    d=$(echo "${f%%.*}")
    unzip $f -d $d;
done

rm *.zip
