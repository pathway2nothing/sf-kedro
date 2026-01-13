#!/bin/bash

conda create -n sf-kedro python=3.12 -y
conda activate sf-kedro
pip install -r requirements.txt
plotly_get_chrome -y
