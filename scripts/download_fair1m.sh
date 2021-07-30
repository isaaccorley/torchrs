pip install gdown
mkdir -p .data/fair1m
gdown --id 1maeEBvno6BhXWXyYqPCug9YVR1yyE-xF
unzip FAIR1M.zip -d .data/fair1m
unzip .data/fair1m/images.zip -d .data/fair1m
unzip .data/fair1m/labelXmls.zip -d .data/fair1m
rm FAIR1M.zip
rm .data/fair1m/images.zip
rm .data/fair1m/labelXmls.zip
