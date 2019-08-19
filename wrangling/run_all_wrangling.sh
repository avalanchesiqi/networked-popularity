#!/bin/bash
# usage: ./run_all_wrangling.sh

log_file=data_wrangling.log

if [ -f "$log_file" ]; then
  rm "$log_file"
fi

python extract_forecast_tsv.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

# I provide the results in ../data/network_pickle so unnecessary to run this script, it takes about 63 hours to finish
# python extract_network_pickle.py >> "$log_file"
#
# sleep 60
# echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python extract_persistent_network.py >> "$log_file"
