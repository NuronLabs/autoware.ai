rm -r build
mkdir build
cd build
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cmake -DCMAKE_PREFIX_PATH="/home/ubuntu/anaconda3/envs/pytorch_latest_p37/lib/python3.7/site-packages/torch/share/cmake/Torch" ..
make

cp ../../model.ts ./
