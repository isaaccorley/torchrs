# Manually download here but requires IEEE login https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection#files
# rehosted to Google Drive to download in script
pip install gdown
mkdir -p .data/oscd
gdown --id 1jidN0DKEIybOrP0j7Bos8bGDDq3Varj3
unzip OSCD.zip -d .data/oscd
rm OSCD.zip
mv '.data/oscd/Onera Satellite Change Detection dataset - Images.zip' .data/oscd/Images.zip
mv '.data/oscd/Onera Satellite Change Detection dataset - Train Labels.zip' .data/oscd/TrainLabels.zip
mv '.data/oscd/Onera Satellite Change Detection dataset - Test Labels.zip' .data/oscd/TestLabels.zip
unzip .data/oscd/Images.zip -d .data/oscd/
unzip .data/oscd/TrainLabels.zip -d .data/oscd/
unzip .data/oscd/TestLabels.zip -d .data/oscd/
mv '.data/oscd/Onera Satellite Change Detection dataset - Images' .data/oscd/images
mv '.data/oscd/Onera Satellite Change Detection dataset - Train Labels' .data/oscd/train_labels
mv '.data/oscd/Onera Satellite Change Detection dataset - Test Labels' .data/oscd/test_labels
rm .data/oscd/Images.zip
rm .data/oscd/TrainLabels.zip
rm .data/oscd/TestLabels.zip
