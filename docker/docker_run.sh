username=yyang
docker run --gpus all -itd --rm -u $(id -u $username):$(id -g $username) --name ${username}_neuraleaf -v /mnt/workspace2024/${username}/NeuraLeaf:/home/${username}/neuraleaf/ repo-luna.ist.osaka-u.ac.jp:5000/${username}/neuraleaf:build bash
