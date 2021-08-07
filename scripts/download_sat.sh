# Converted from mat to hdf5 format for better i/o support
mkdir -p .data/sat
pip install gdown
gdown --id 1q4Xpi67DQtbLx1tnA9XZGfvBRIXYr8dG
unzip Sat.zip -d .data/sat
rm Sat.zip
