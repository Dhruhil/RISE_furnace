"rise_furnace_mid_part_base_case_coarse.geo" is the script used to generate the geometry in Gmesh. 
"rise_furnace_mid_part_base_case_coarse.msh" is the mesh built from the corresponding .geo file (what you get after using the
meshing function in Gmesh).

"rise_furnace_mid_part_base_case_refined_#.geo" are scripts used to generate refined base_case_coarse meshes in Gmesh.  
"rise_furnace_mid_part_base_case_refined_#.msh" are the meshes built from the corresponding .geo files.

The .msh files, which must be exported from Gmsh on Version 2 ASCII format (with "Save all elements" and "Save parametric coordinates" unchecked) can be converted to FOAM format using gmshToFoam <.msh file> Thereafter, the mesh needs to be split up into its different regions (if it consists of more than one region) using splitMeshRegions -cellZones -overwrite (note that filename is not needed at the end here)

After the split, don't forget to go into constant/inner_box/polyMesh/boundary and change "inGroups" in the file "boundary" from 1(wall) to 2(wall viewFactorWall), to make the inner box walls irradiating.
For this, you can use the "edit_files.sh" script.

"fvOptions" in the "system/<region>" folders contains controls for the heating element powers.

With the "remove_ts_folders.sh" script, you can remove all saved time step folders except the "0" folder.

 
