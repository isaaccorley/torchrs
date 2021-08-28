mkdir -p .data/idtrees
wget https://zenodo.org/record/3934932/files/IDTREES_competition_train_v2.zip?download=1 -O IDTREES_competition_train_v2.zip
wget https://zenodo.org/record/3934932/files/IDTREES_competition_test_v2.zip?download=1 -O IDTREES_competition_test_v2.zip
unzip IDTREES_competition_train_v2.zip -d .data/idtrees
unzip IDTREES_competition_test_v2.zip -d .data/idtrees
rm IDTREES_competition_train_v2.zip
rm IDTREES_competition_test_v2.zip
