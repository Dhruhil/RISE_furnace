#!/bin/bash

read -p "What is the last timestep? " i

for n in $(seq 1 "$i"); do
    rm -rf "$n"
done
