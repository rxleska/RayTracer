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
    - [ ] png/image mapped materials (child of Lambertian)
- [X] Camera
    - [X] positionable
    - [X] focusable
    - [X] aperture
- [X] Multi-threading
- [ ] Optimization Data Structures
    - [ ] Octree 
- [X] hittable objects
    - [X] Sphere
    - [X] Polygons (imo this is better than creating quadralaterals if you have a quadtree or another spatial partitioning method) (also look into more than 3 index polygons)

- [ ] other ideas
    - [ ] live render preview 
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