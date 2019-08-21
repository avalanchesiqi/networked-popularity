#!/bin/bash
# usage: ./run_all_measures.sh

log_file=measures.log

if [ -f "$log_file" ]; then
  rm "$log_file"
fi

python plot_fig3_rel2rec.py.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig4_basic_statistics.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig5_how_videos_connect.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

## I provide the result 'bowtie_cutoff.log' so unnecessary to run this script, it takes about 2 hours to finish
# python how_bowtie_changes_with_cutoff.py >> "$log_file"
python plot_fig7_bowtie_changes_with_cutoff.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig8_yearly_spearmanr.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

## I provide the result 'bowtie_evolves.log' so unnecessary to run this script, it takes about 3 hours to finish
# python python how_bowtie_evolves.py.py >> "$log_file"
python plot_fig9_bowtie_evolves.py >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python plot_fig10_temporal_micro.py >> "$log_file"
