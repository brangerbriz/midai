#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! $(which wget) ]
then
	echo 'Please install wget. If on Ubuntu, run "sudo apt-get install wget".'
fi

exit 0

cd $DIR/features/music21

echo "Downloading LMD features..."
wget "https://github.com/brangerbriz/t-SNEPointSelector/releases/download/data/lmd_features.csv.tar.gz"
echo "Downloading LMD mono features..."
wget "https://github.com/brangerbriz/t-SNEPointSelector/releases/download/data/lmd_mono_tracks_features.csv.tar.gz"

echo "Untaring..."
tar xzf "lmd_mono_tracks_features.csv.tar.gz"
tar xzf "lmd_features.csv.tar.gz"

echo "Removing tarballs..."
rm "lmd_mono_tracks_features.csv.tar.gz"
rm "lmd_features.csv.tar.gz"

cd ../../midi

echo "Downloading LMD files..."
wget "https://github.com/brangerbriz/t-SNEPointSelector/releases/download/data/lmd_midi.tar.gz"
echo "Downloading LMD mono files..."
wget "https://github.com/brangerbriz/t-SNEPointSelector/releases/download/data/lmd_mono_tracks_seperated.tar.gz"

echo "Untaring..."
tar xzf "lmd_midi.tar.gz"
tar xzf "lmd_mono_tracks_seperated.tar.gz"

echo "Removing tarballs..."
rm "lmd_midi.tar.gz"
rm "lmd_mono_tracks_seperated.tar.gz"

cd ..

echo "Done."