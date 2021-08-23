sudo apt install p7zip-full
mkdir -p .data/inria_ail
wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001
wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002
wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003
wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004
wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005
7z x aerialimagelabeling.7z.001
unzip NEW2-AerialImageDataset.zip -d .data/
rm -i aerialimagelabeling.7z.*
rm -i NEW2-AerialImageDataset.zip
