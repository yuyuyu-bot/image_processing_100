# image_processing_100

[画像処理100本ノック](https://github.com/yoyoyo-yo/Gasyori100knock)を
- C++
- NEON
- CUDA

でやる。

# Set up history
```shell-session
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda


sudo apt install git cmake clang build-essential libc6-dbg
export PATH=${PATH}:/usr/local/cuda-12.0/bin/
sudo apt install -y libopengl-dev libegl1 libxdamage1
```

# Perf
https://gist.github.com/abel0b/b1881e41b9e1c4b16d84e5e083c38a13
```shell-session
sudo apt install -y flex bison
git clone https://github.com/microsoft/WSL2-Linux-Kernel --depth 1
cd WSL2-Linux-Kernel/tools/perf
make -j8
sudo cp perf /usr/local/bin

git clone https://github.com/brendangregg/FlameGraph
```

run
```
echo 1 | sudo tee /proc/sys/kernel/sched_schedstats
perf record -- <command>
perf report | head -n 20
perf script | ~/git/FlameGraph/stackcollapse-perf.pl > out.perf-folded
echo 0 | sudo tee /proc/sys/kernel/sched_schedstats

export PERF_EXEC_PATH=/usr/local/bin/

```
