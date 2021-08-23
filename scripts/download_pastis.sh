mkdir -p .data/pastis
wget --no-check-certificate https://zenodo.org/record/5012942/files/PASTIS.zip?download=1 -O PASTIS.zip
unzip PASTIS.zip -d .data/pastis/
rm PASTIS.zip
