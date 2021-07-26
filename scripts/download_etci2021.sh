pip install gdown
mkdir -p .data/etci2021

# train dataset (3GB)
gdown --id 14HqNW5uWLS92n7KrxKgDwUTsSEST6LCr
unzip train.zip -d .data/etci2021
rm -r .data/etci2021/__MACOSX
rm train.zip

# val dataset (0.85GB)
gdown --id 19sriKPHCZLfJn_Jmk3Z_0b3VaCBVRVyn
unzip val_with_ref_labels.zip -d .data/etci2021
rm -r .data/etci2021/__MACOSX
rm val_with_ref_labels.zi

# test dataset (1.2GB) (no labels)
gdown --id 1rpMVluASnSHBfm2FhpPDio0GyCPOqg7E
unzip test_without_ref_labels.zip -d .data/etci2021
rm -r .data/etci2021/__MACOSX
rm test_without_ref_labels.zip
