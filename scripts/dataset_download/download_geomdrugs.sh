#!\bin\bash
cd $1

wget https://dataverse.harvard.edu/api/access/datafile/4327252

md5sum 4327252
tar -xf 4327252
mv rdkit_folder geom_drugs
