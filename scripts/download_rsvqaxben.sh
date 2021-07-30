mkdir -p .data/rsvqaxben/
wget https://zenodo.org/record/5084904/files/LRBENanswers.json?download=1 -O .data/rsvqaxben/LRBENanswers.json
wget https://zenodo.org/record/5084904/files/LRBENimages.json?download=1 -O .data/rsvqaxben/LRBENimages.json
wget https://zenodo.org/record/5084904/files/LRBENpeople.json?download=1 -O .data/rsvqaxben/LRBENpeople.json
wget https://zenodo.org/record/5084904/files/LRBENquestions.json?download=1 -O .data/rsvqaxben/LRBENquestions.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_test_answers.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_test_answers.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_test_images.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_test_images.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_test_questions.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_test_questions.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_train_answers.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_train_answers.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_train_images.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_train_images.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_train_questions.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_train_questions.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_val_answers.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_val_answers.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_val_images.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_val_images.json
wget https://zenodo.org/record/5084904/files/RSVQAxBEN_split_val_questions.json?download=1 -O .data/rsvqaxben/RSVQAxBEN_split_val_questions.json
wget https://zenodo.org/record/5084904/files/Images.zip?download=1 -O .data/rsvqaxben/Images.zip
unzip .data/rsvqaxben/Images.zip -I .data/rsvqaxben/
rm .data/rsvqaxben/Images.zip
