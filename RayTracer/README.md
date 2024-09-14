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
- [ ] hittable objects
    - [ ] Sphere
    - [ ] Polygons (imo this is better than creating quadralaterals if you have a quadtree or another spatial partitioning method) (also look into more than 3 index polygons)

- [ ] other ideas
    - [ ] live render preview 
    - [ ] dynamic sample rate anti-aliasing (edges of objects send signal to increase sample rate like msaa)