#!/bin/bash
#PBS -l select=5:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:30:00
#PBS -q <queue>
#PBS -A <account>
#PBS -o train.out
#PBS -e train.err

nodes=4
gpus_per_node=4

# should the training utilize GPU-index-batching
allGPU=True

# which dataset to train on; valid options include "pems-bay", 'pemsAllLA', and "pems"
dataset="pems"

# total workers
total=$((gpus_per_node * nodes))

readarray -t all_nodes < "$NODEFILE"

# use first node for dask scheduler and client
scheduler_node=${all_nodes[0]}
monitor_node=${all_nodes[1]}

# all nodes but first for workers
tail -n +2 $NODEFILE > worker_nodefile.txt

echo "Launching scheduler"
mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node dask scheduler --scheduler-file cluster.info &
scheduler_pid=$!

# wait for the scheduler to generate the cluster config file
while ! [ -f cluster.info ]; do
    sleep 1
    echo .
done

echo "$total workers launching" 
mpiexec -n $total --ppn $gpus_per_node --cpu-bind none --hostfile worker_nodefile.txt dask worker --local-directory /local/scratch --scheduler-file cluster.info --nthreads 8 --memory-limit 512GB &

# give workers a bit to launch
sleep 5

echo "Launching client"
mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` pems_ddp.py --dask-cluster-file cluster.info -np $gpus_per_node -g $allGPU --dataset $dataset
