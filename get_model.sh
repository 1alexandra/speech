#!/bin/sh

mkdir -p models
fileid="1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6"
filename="models/vggish_audioset_weights.h5"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
