#!/bin/bash
for i in {1..8}; do
	sed -i "s/1.2e7/2.9e6/g" heater_$i/fvOptions
done
