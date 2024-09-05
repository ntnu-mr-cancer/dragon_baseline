#!/usr/bin/env bash

./build.sh

docker save joeranbosma/dragon_baseline:latest | gzip -c > dragon_baseline.tar.gz
