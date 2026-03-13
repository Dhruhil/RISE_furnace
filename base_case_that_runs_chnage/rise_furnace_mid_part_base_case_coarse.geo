// Gmsh project created on Fri May 16 21:00:25 2025
SetFactory("OpenCASCADE");
//+
Disk(1) = {0.0325, 0.0275, 0, 0.0175, 0.0175};
//+
Translate {0.047, 0, 0} {
  Duplicata { Point{1}; Curve{1}; Surface{1}; }
}
//+
Translate {0.047, 0, 0} {
  Duplicata { Point{2}; Curve{2}; Surface{2}; }
}
//+
Translate {0.047, 0, 0} {
  Duplicata { Point{3}; Curve{3}; Surface{3}; }
}
//+
Translate {0, 0.305, 0} {
  Duplicata { Point{4}; Point{3}; Point{2}; Point{1}; Curve{4}; Curve{3}; Curve{2}; Curve{1}; Surface{4}; Surface{3}; Surface{2}; Surface{1}; }
}
//+
Box(1) = {0, 0, 0, 0.206, 0.36, 0.39};
//+
Transfinite Curve {4, 3, 2, 1, 9, 10, 11, 12} = 10 Using Progression 1;
//+
Extrude {0, 0, 0.39} {
  Surface{4}; Curve{4}; Curve{3}; Surface{3}; Surface{2}; Curve{2}; Surface{1}; Curve{1}; Surface{5}; Curve{9}; Surface{6}; Curve{10}; Surface{7}; Curve{11}; Surface{8}; Curve{12}; Layers {10}; 
}
//+
Transfinite Curve {26, 28, 30, 32, 34, 36, 38, 40, 40} = 10 Using Progression 1;
//+
BooleanFragments{ Volume{2}; Volume{3}; Volume{4}; Volume{5}; Volume{1}; Volume{9}; Volume{8}; Volume{7}; Volume{6}; Delete; }{ }
//+
// === CYLINDER_DISK_START ===
Disk(45) = {0, 0.18, 0.195, 0.05, 0.05};
// === CYLINDER_DISK_END ===

//+
Rotate {{0, 1, 0}, {0, 0, 0.195}, Pi/2} {
  Curve{69}; Surface{45}; 
}//+

// === CYLINDER_EXTRUDE_START ===
Extrude {0.1, 0, 0} {
  Curve{69}; Surface{45}; Layers {10}; 
}
// === CYLINDER_EXTRUDE_END ===

//+
BooleanFragments{ Volume{2}; Volume{3}; Volume{4}; Volume{5}; Volume{11}; Volume{10}; Volume{9}; Volume{8}; Volume{7}; Volume{6}; Delete; }{ }
//+
Box(13) = {-0.04, 0.065, 0, 0.04, 0.23, 0.39};
//+
BooleanFragments{ Volume{2}; Volume{12}; Volume{11}; Volume{13}; Volume{9}; Volume{8}; Volume{7}; Volume{6}; Volume{3}; Volume{4}; Volume{5}; Delete; }{ }
//+
Transfinite Curve {69, 71} = 20	 Using Progression 1;
//+
Box(14) = {-0.14, -0.1, -0.08, 0.486, 0.56, 0.55};
//+
BooleanFragments{ Volume{13}; Volume{11}; Volume{12}; Volume{9}; Volume{8}; Volume{7}; Volume{6}; Volume{2}; Volume{3}; Volume{4}; Volume{5}; Volume{14}; Delete; }{ }
//+
Box(15) = {-0.057, 0, -0.19, 0.32, 0.36, 0.11};
//+
BooleanFragments{ Volume{15}; Volume{13}; Volume{11}; Volume{12}; Volume{2}; Volume{3}; Volume{4}; Volume{5}; Volume{9}; Volume{8}; Volume{7}; Volume{6}; Volume{14}; Delete; }{ }
//+
Physical Volume("outer_box", 136) = {14, 15};
//+
Physical Volume("inner_box", 137) = {12};
//+
Physical Volume("brick_heater", 138) = {13};
//+
Physical Volume("steel_cylinder", 139) = {11};
//+
Physical Volume("heater_1", 140) = {9};
//+
Physical Volume("heater_2", 141) = {8};
//+
Physical Volume("heater_3", 142) = {7};
//+
Physical Volume("heater_4", 143) = {6};
//+
Physical Volume("heater_5", 144) = {5};
//+
Physical Volume("heater_6", 145) = {4};
//+
Physical Volume("heater_7", 146) = {3};
//+
// Physical Volume("heater_8", 147) = {3};
//+
// Physical Volume(" heater_8", 147) -= {3};
//+
Physical Volume("heater_8", 147) = {2};
