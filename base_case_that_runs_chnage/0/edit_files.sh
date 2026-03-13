#!/bin/bash
for i in {1..8}; do
	sed -i "s/heater_1/heater_$i/g" heater_$i/T
done
