// Uniform sizing: isolate geometry/meshing issues from the field
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.MeshSizeMin = 5.0;
Mesh.MeshSizeMax = 5.0;

Mesh.Algorithm = 6; // Frontal-Delaunay