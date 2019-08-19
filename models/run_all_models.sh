#!/bin/bash
# usage: ./run_all_models.sh

log_file=models.log

if [ -f "$log_file" ]; then
  rm "$log_file"
fi

python justify_persistent_link.py >> "$log_file"
python compute_linkage_pearsonr.py >> "$log_file"
python plot_fig11_profile_persistent_network.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig4_basic_statistics.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig5_how_videos_connect.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"
