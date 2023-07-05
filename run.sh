#!/bin/bash




CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate flower


nc=3
K=1
gamma=1

python run_server.py --nc $nc --K $K --gamma $gamma 

#echo "starting clients"
#for ((i = 0; i<=$nc-1; i++))
#do
#    sleep 1

#    python run_client.py --cid $i --K $K --gamma $gamma &
#done

#wait $!

#exit