pip install gdown
mkdir -p .data/
gdown --id 1JamSsxiytXdzAIk6VDVWfc-OsX-81U81
unzip LEVIR-CD+.zip -d .data/
mv .data/LEVIR-CD+ .data/levircd_plus
rm -r .data/__MACOSX
find .data/levircd_plus/ -name '.DS_Store' -type f -delete
rm LEVIR-CD+.zip
