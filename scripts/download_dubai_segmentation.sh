mkdir -p .data
wget https://semantic-segmentation-uae.s3.eu-central-1.amazonaws.com/Semantic-segmentation-dataset-1.zip
unzip Semantic-segmentation-dataset-1.zip
mv "Semantic segmentation dataset" dubai-segmentation
mv dubai-segmentation .data/
rm Semantic-segmentation-dataset-1.zip
