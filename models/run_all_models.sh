#!/bin/bash
# usage: ./run_all_models.sh

log_file=models.log

if [ -f "$log_file" ]; then
  rm "$log_file"
fi

## I provide the result 'justify_persistent_link.log' so unnecessary to run this script, it takes about 2 hours to finish
# python justify_persistent_link.py >> "$log_file"
## I provide the result 'random_pearsonr.log', 'ephemeral_pearsonr.log', 'persistent_pearsonr.log',
## 'reciprocal_pearsonr.log', so unnecessary to run this script, it takes about 2 hours to finish
# python compute_linkage_pearsonr.py >> "$log_file"
python plot_fig11_profile_persistent_network.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

## I provide the result 'forecast_tracker_all.json' so unnecessary to run this script
# python forecast_next_week.py >> "$log_file"
python plot_fig4_basic_statistics.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig5_how_videos_connect.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"
