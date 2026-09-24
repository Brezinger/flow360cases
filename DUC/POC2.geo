Mesh.MaxNumThreads2D = 1;

Field[1] = AttractorAnisoCurve;
Field[1].CurvesList = {407, 416, 420};
Field[1].Sampling = 1000;

// Sizes close to the curves:
Field[1].SizeMinNormal  = 0.1;
Field[1].SizeMinTangent = 3.0;

// Sizes far from the curves:
Field[1].SizeMaxNormal  = 3.0;
Field[1].SizeMaxTangent = 3.0;

// Distance range for the transition:
Field[1].DistMin = 1.0;
Field[1].DistMax = 20.0;

Background Field = 1;

Mesh.Algorithm = 6; // Frontal-Delaunay default
MeshAlgorithm Surface {95} = 7; // BAMG: anisotropic triangles
