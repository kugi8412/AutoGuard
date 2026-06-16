#!/bin/bash

FOLDER_URL="https://drive.google.com/drive/folders/1o0jM55cw5XEY36brsrWEb2rBQMe_StZE?usp=sharing"
TARGET_DIR="data"

echo "Creating target directory '$TARGET_DIR'."
mkdir -p "$TARGET_DIR"

if ! command -v gdown &> /dev/null
then
    pip install gdown
fi

# Google Drive download
gdown --folder "$FOLDER_URL" -O "$TARGET_DIR/" --continue

# Flatten the directory structure
if [ -d "$TARGET_DIR/data_autoguard" ]; then
    echo "Extracting files from 'data_autoguard' directly into '$TARGET_DIR'."
    mv "$TARGET_DIR/data_autoguard/"* "$TARGET_DIR/" 2>/dev/null
    mv "$TARGET_DIR/data_autoguard/".[!.]* "$TARGET_DIR/" 2>/dev/null
    rmdir "$TARGET_DIR/data_autoguard"
fi

echo "[SUCCESS]: All files are now directly in '$TARGET_DIR'."
