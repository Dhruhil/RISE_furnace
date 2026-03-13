#!/bin/bash
for i in {1..8}; do
	sed -i "s/heating_element_1/heater_$i/g" heater_$i/boundaryRadiationProperties
done
