username=yyang
project_name=neuraleaf
container_name=neuraleaf
folder_name=NeuraLeaf

# Run Docker container with:
# - GPU support
# - Project directory mount (includes results/, checkpoints/, etc. for TensorBoard logs)
# - Dataset directory mount
# - X11 display for GUI visualization
# - TensorBoard port mapping (6006) - access at http://localhost:6006
# - Visdom port mapping (8097) - access at http://localhost:8097
# - Increased shared memory for PyTorch DataLoader
docker run --gpus all -itd \
    -u $(id -u $username):$(id -g $username) \
    --name ${username}_${container_name} \
    -v /mnt/workspace2024/${username}/${folder_name}:/home/${username}/mnt/workspace \
    --mount type=bind,source="/mnt/poplin/share/2023/users/yang/NeuraLeaf_dataset",target=/mnt/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 6006:6006 \
    -p 8097:8097 \
    --shm-size=8g \
    repo-luna.ist.osaka-u.ac.jp:5000/${username}/${project_name}:build \
    bash
