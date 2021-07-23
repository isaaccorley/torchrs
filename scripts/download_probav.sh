mkdir -p .data/probav
wget --no-check-certificate https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip
unzip probav_data.zip -d .data/probav
rm probav_data.zip
