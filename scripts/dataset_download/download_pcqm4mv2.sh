#!\bin\bash


cd $1 
mkdir pcqm_raw
cd pcqm_raw
wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
md5sum pcqm4m-v2-train.sdf.tar.gz # fd72bce606e7ddf36c2a832badeec6ab
tar -xf pcqm4m-v2-train.sdf.tar.gz

