mkdir -p .data/rsicd

# Download images
pip install gdown
apt-get install unrar
gdown --id 0B1jt7lJDEXy3SmZEdDd0aWpXcWc
unrar x RSICD_images.rar .data/rscid
rm RSICD_images.rar

# Download annotations
gdown --id 1q8EcBWuCbvtTnMILE60WOhd9S0C4rsfT
mv dataset_rsicd.json .data/rsicd/
