rm /dev/shm/cuda_dedup_shared
python ../pytorch-benchmarks/main.py -eo -ef /root/dataset/cifar10_folder &
sleep 3
python ../pytorch-benchmarks/main.py -eo -ef /root/dataset/cifar10_folder
# /mnt/data/dataset/cifar10_folder