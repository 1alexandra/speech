# Simple speaker recognition project

Sources:

SPEAKER IDENTIFICATION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS https://habr.com/ru/post/144491/

ATTENTION-BASED MODELS FOR TEXT-DEPENDENT SPEAKER VERIFICATION https://arxiv.org/pdf/1710.10470.pdf

Dataset: 

VoxCeleb http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

### How to use:

1. clone repo:
```
git clone https://github.com/1alexandra/speech.git
cd speech
```

2. load dataset:
```
cd data
sh ./get_data.sh
unzip vox1_dev_wav.zip
cd ../
```

3. extract mel features:
```
python cook_data.py
```

4. if you don't wanna to train VGGish, load pretrained weights for it [here](https://drive.google.com/open?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6) and place it to `./models`

...or you can simply run this command:
```
sh ./get_model.sh
```

5. ...

6. PROFIT!!
