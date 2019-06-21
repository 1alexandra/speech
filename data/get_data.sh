#!/bin/sh

user=voxceleb1904
pass=9hmp7488
prefix="http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta"
for part in a b c d; do
    url=$prefix$part;
    wget $url --user=$user --password=$pass
done
cat vox1_dev* > vox1_dev_wav.zip
rm vox1_dev_wav_part*

