# From Scratch Version of a Ray Tracer in C++ 

## Goal
The goal of this subsection is to create a ray tracer from scratch in C++. I will not look at any code from the textbook during development. I will only use the textbook as a reference for the theory behind the code and for mathematic formulas. 

## Subgoals differing from textbook implementation

### Subgoal 1 implement MSAA edge detection and anti-aliasing
    More information here


### Subgoal 2 implement a quadtree (Octree since 3d) for spatial partitioning (instead of bounding box hierarchies)
    More information here


### Subgoal 3 - TBD - need more ideas
    More information here 


## Features
- [X] Basic ray tracing
- [-] Materials
    - [X] Lambertian
    - [X] Metal
    - [X] Dielectric
    - [X] Diffuse light
    - [X] png/image mapped materials (child of Lambertian)
    - [ ] Volume Materials (smoke, clouds, etc.)
- [X] Camera
    - [X] positionable
    - [X] focusable
    - [X] aperture
- [X] Multi-threading
    - [X] Using Cuda for GPU acceleration
- [X] Optimization Data Structures
    - [X] Octree 
- [X] hittable objects
    - [X] Sphere
    - [X] Polygon_Ts (imo this is better than creating quadralaterals if you have a quadtree or another spatial partitioning method) (also look into more than 3 index Polygon_Ts)

- [ ] other ideas
    - [ ] live render preview (probably not doing this one) 
    - [X] dynamic sample rate anti-aliasing (edges of objects send signal to increase sample rate like msaa)
    - [ ] Perlin noise



## Assignment Requirements
To earn a grade of at least C- in this course, your ray tracer must include this basic functionality:

    A camera with configurable position, orientation, and field of view
    Anti-aliasing
    Ray/sphere intersections
    Ray/triangle intersections
    The ability to load textures (file format(s) of your choice; may use third-party libraries)
    Textured spheres and triangles
    The ability to load and render triangle meshes (file format(s) of your choice; may use third-party libraries for loading)
    A spatial subdivision acceleration structure of your choice
    Specular, diffuse, and dielectric materials (per first volume of Ray Tracing in One Weekend series)
    Emissive materials (lights)

In addition, for a C- you must implement at least X points worth of features from the following list (point values coming soon; these range from very easy to fairly complex):

    High dynamic range images
    Volume rendering (smoke, clouds, etc.)
    Quads
    Quadrics
    Spectral rendering
    BRDF materials (Bi-directional reflectance distribution functions)
    Subsurface scattering (BSSRDFs)
    Motion blur
    Defocus blur/depth of field
    Object instancing
    Perlin noise
    Cube maps
    Importance sampling
    Round pixels
    Other features you discover through reading, YouTube, etc. (tell me about them and they'll be added to this list explicitly)




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