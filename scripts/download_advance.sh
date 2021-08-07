mkdir -p .data/advance
wget --no-check-certificate https://zenodo.org/record/3828124/files/ADVANCE_vision.zip?download=1 -O ADVANCE_vision.zip
wget --no-check-certificate https://zenodo.org/record/3828124/files/ADVANCE_sound.zip?download=1 -O ADVANCE_sound.zip
unzip ADVANCE_vision.zip -d .data/advance/
rm ADVANCE_vision.zip
unzip ADVANCE_sound.zip -d .data/advance/
rm ADVANCE_sound.zip
