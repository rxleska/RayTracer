# Ray Tracer in CUDA C++ 

## Goal
The goal of this subsection is to create a ray tracer from scratch in CUDA for C++.

## Features

- Required 
    - Camera Configuration
        - Position, Orientation, Field of View
    - Anti-Aliasing
        - configurable sample count 
        - See also Adaptive Sampling
    - Ray/Sphere Intersection
    - Ray/Triangle Intersection (Polygon)
    - Load texture (PPM format)
    - Textured Spheres and Triangle (both have u,v mapping)
    - Acceleration Datastructure (Octree)
    - Specular (Metal), Diffuse (Lambertian), and Dielectric Materials 
    - Emissive Material (Light)
- For Points
    - HDR (High Dynamic Range) 
    - Volume Rendering (smoke, clouds)
    - Quads (Rectangle, Box)
    - Motion Blur 
    - Defocus/Depth of Field (see camera configuration)
    - Object Instancing (Translation, Rotation)
    - Importance Sampling (Cosine weighted and Light Sampling)
    - Round Pixels 
    - Parallelization (CUDA)
    - GPU Acceleration (CUDA)
    - Adaptive Sampling (MSAA)   
- For Fun
    - Phong lighting model 
    - Phong-Lambertian Material
    - LambertianBordered Material
    - Other Sampling Methods (Beckmann, Blinn-Phong, Square, Uniform)  


## MESH LICENSES

### Person Model
Model Information:
* title:	Low-poly Male Base Mesh
* source:	https://sketchfab.com/3d-models/low-poly-male-base-mesh-7b9411ff0c624321baf4caac014bc0bd
* author:	decodigo (https://sketchfab.com/decodigo)

Model License:
* license type:	CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
* requirements:	Author must be credited. Commercial use is allowed.

If you use this 3D model in your project be sure to copy paste this credit wherever you share it:
This work is based on "Low-poly Male Base Mesh" (https://sketchfab.com/3d-models/low-poly-male-base-mesh-7b9411ff0c624321baf4caac014bc0bd) by decodigo (https://sketchfab.com/decodigo) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)



### Knight Model
Model Information:
* title:	Low Poly Chess Knight (Unpainted)
* source:	https://sketchfab.com/3d-models/low-poly-chess-knight-unpainted-5cd4255a6db54f9e9867bfbcfef65f34
* author:	Justin Vaughn (https://sketchfab.com/jvaughn)

Model License:
* license type:	CC-BY-NC-4.0 (http://creativecommons.org/licenses/by-nc/4.0/)
* requirements:	Author must be credited. No commercial use.

If you use this 3D model in your project be sure to copy paste this credit wherever you share it:
This work is based on "Low Poly Chess Knight (Unpainted)" (https://sketchfab.com/3d-models/low-poly-chess-knight-unpainted-5cd4255a6db54f9e9867bfbcfef65f34) by Justin Vaughn (https://sketchfab.com/jvaughn) licensed under CC-BY-NC-4.0 (http://creativecommons.org/licenses/by-nc/4.0/)