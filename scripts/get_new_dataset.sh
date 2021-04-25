wget "https://github.com/NotJoeMartinez/APDL_Quality_Control/blob/main/datasets/original.tar.gz?raw=true"
mv 'original.tar.gz?raw=true' original.tar.gz
tar -zxvf original.tar.gz
find original -type f -name '\.*' -delete
