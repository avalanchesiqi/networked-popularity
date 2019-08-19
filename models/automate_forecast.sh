#!/usr/bin/env bash

for i in $(seq 1 10)
do
  python -u forecast_next_week.py $i 2>&1

  sleep 30
  echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
done
