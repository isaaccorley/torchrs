apt-get install -y p7zip-full
mkdir -p .data/s2mtcp
wget --no-check-certificate https://zenodo.org/record/4280482/files/S2MTCP_data.7z?download=1 -O S2MTCP_data.7z
wget --no-check-certificate https://zenodo.org/record/4280482/files/S2MTCP_metadata.csv?download=1 -O .data/s2mtcp/S2MTCP_metadata.csv
wget https://zenodo.org/record/4280482/files/README.txt?download=1 -O .data/s2mtcp/README.txt
7z x S2MTCP_data.7z -o.data/s2mtcp
rm S2MTCP_data.7z