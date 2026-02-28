sudo apt-get update
sudo apt-get install -y runc uidmap

cd ~
wget https://github.com/moby/buildkit/releases/download/v0.28.0-rc2/buildkit-v0.28.0-rc2.linux-amd64.tar.gz
tar xvfz buildkit-v0.28.0-rc2.linux-amd64.tar.gz
sudo mv bin/* /usr/local/bin/
rm -rf bin buildkit-v0.28.0-rc2.linux-amd64.tar.gz


export BUILDKIT_HOST=unix:///run/buildkit/buildkitd.sock


curl -sSL https://github.com/docker/buildx/releases/download/v0.31.1/buildx-v0.31.1.linux-amd64 -o buildx

# Make it executable
chmod +x buildx
sudo mv buildx /usr/local/bin/buildx

sudo buildx create --use --name standalone-builder --driver remote unix:///run/buildkit/buildkitd.sock
