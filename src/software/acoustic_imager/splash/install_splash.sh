#!/usr/bin/env bash
set -e

THEME_NAME="mytheme"
THEME_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THEME_DEST_DIR="/usr/share/plymouth/themes/${THEME_NAME}"

echo "Installing Acoustic Imager splash theme..."

# Ensure plymouth is installed
if ! dpkg -s plymouth >/dev/null 2>&1; then
    echo "Installing plymouth..."
    sudo apt update
    sudo apt install -y plymouth plymouth-themes
fi

# Create destination directory
sudo mkdir -p "$THEME_DEST_DIR"

# Copy theme files
sudo cp "${THEME_SRC_DIR}/splash.png" "$THEME_DEST_DIR/"
sudo cp "${THEME_SRC_DIR}/mytheme.plymouth" "$THEME_DEST_DIR/"
sudo cp "${THEME_SRC_DIR}/mytheme.script" "$THEME_DEST_DIR/"

# Register theme
sudo update-alternatives --install \
  /usr/share/plymouth/themes/default.plymouth \
  default.plymouth \
  "${THEME_DEST_DIR}/mytheme.plymouth" 100

sudo update-alternatives --set default.plymouth \
  "${THEME_DEST_DIR}/mytheme.plymouth"

# Rebuild initramfs
sudo update-initramfs -u

echo "Splash theme installed successfully."
echo "Reboot to see changes."