curl -O https://digip.org/jansson/releases/jansson-2.13.tar.bz2
bunzip2 -c jansson-2.13.tar.bz2 | tar xf -
rm jansson-2.13.tar.bz2
cd jansson-2.13
./configure
make
sudo make install
cd ..
