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
python extract_mel.py
```

4. ...

5. PROFIT!!
