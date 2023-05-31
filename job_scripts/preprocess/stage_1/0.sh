#!/bin/bash
#SBATCH --time=07:30:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2

./rclone/rclone copy -P trecbox:Spotify-Podcasts/EN/podcasts-audio-only-2TB/podcasts-audio ./spotify-podcasts
