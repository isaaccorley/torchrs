mkdir -p .data/
wget https://cloud.sylvainlobry.com/s/4Qg5AXX8YfCswmX/download -O RSVQA_LR.zip
unzip RSVQA_LR.zip -d .data/
unzip .data/RSVQA_LR/Images_LR.zip -d .data/RSVQA_LR
rm RSVQA_LR.zip
rm .data/RSVQA_LR/Images_LR.zip
