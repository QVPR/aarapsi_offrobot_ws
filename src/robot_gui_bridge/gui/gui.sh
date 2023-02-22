#!/bin/bash
# https://stackoverflow.com/questions/24112727/relative-paths-based-on-file-location-instead-of-current-working-directory
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
cd "${scriptDir}"
pwd
#http-server -p 9090 -a 172.19.31.76 -d false -i false -o gui.html -d "${scriptDir}"
python -m http.server 9090 -b 172.19.31.76
