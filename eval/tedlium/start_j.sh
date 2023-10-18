#!/bin/bash

for VARIABLE in {1..200}
	do
		sbatch beam_search_grid_64.sh
	done


