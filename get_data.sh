#!/bin/bash

FOLDER_URL="https://drive.google.com/drive/folders/1o0jM55cw5XEY36brsrWEb2rBQMe_StZE?usp=sharing"
TARGET_DIR="data"

echo "Creating target directory '$TARGET_DIR'."
mkdir -p "$TARGET_DIR"

if ! command -v gdown &> /dev/null
then
    echo "Installing gdown."
    pip install gdown
fi

echo "Starting Google Drive download."

if gdown --folder "$FOLDER_URL" -O "$TARGET_DIR/" --continue --remaining-ok; then
    
    # If gdown succeeds, flatten the directory structure
    if [ -d "$TARGET_DIR/data_autoguard" ]; then
        echo "Extracting files from 'data_autoguard' directly into '$TARGET_DIR'."
        mv "$TARGET_DIR/data_autoguard/"* "$TARGET_DIR/" 2>/dev/null
        mv "$TARGET_DIR/data_autoguard/".[!.]* "$TARGET_DIR/" 2>/dev/null
        rmdir "$TARGET_DIR/data_autoguard" 2>/dev/null
    fi

    echo "[SUCCESS]: All files are now directly in '$TARGET_DIR'."

else
    # FALLBACK: If gdown fails
    echo ""
    echo "====================================================================="
    echo "[ERROR]: Automatic download failed (likely due to Google Drive rate limits)."
    echo "[FALLBACK]: Please download the files manually by following these instructions:"
    echo ""
    echo "1. Copy and paste this link into your web browser:"
    echo "   $FOLDER_URL"
    echo "2. Click 'Download all' (Google will automatically create a .zip file)."
    echo "3. Extract the archive into the '$TARGET_DIR' folder using the command:"
    echo "   unzip autogurad_data.zip -d $TARGET_DIR/"
    echo "====================================================================="
    exit 1
fi
