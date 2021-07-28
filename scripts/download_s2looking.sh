apt-get install subversion
mkdir -p .data/
svn export https://github.com/AnonymousForACMMM/Dataset/trunk/S2Looking/S2Looking/train/ .data/s2looking/train/
svn export https://github.com/AnonymousForACMMM/Dataset/trunk/S2Looking/S2Looking/val/ .data/s2looking/val/
svn export https://github.com/AnonymousForACMMM/Dataset/trunk/S2Looking/S2Looking/test/ .data/s2looking/test/
