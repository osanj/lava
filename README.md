# Lava

A highlevel wrapper of Vulkan's compute API with numpy bridge.

numpy + shaders = <3

### To-Do List
- [ ] memory alignment of (arbitrary) complex types
  - [ ] scalars
    - [x] to shader
    - [ ] from shader
  - [ ] vectors
    - [x] to shader
    - [ ] from shader
  - [ ] matrices
    - [ ] to shader
    - [ ] from shader
  - [ ] structures
    - [x] to shader
    - [ ] from shader
  - [ ] multi-dimensional arrays
    - [x] to shader
    - [ ] from shader
  - [ ] dynamic length
- [ ] ~~images / samplers~~ (for now)
- [ ] push constants
- [ ] bytecode analysis
  - [ ] bindings (std140/std430, variable?)
  - [x] local groups (?)
  - [x] misc (glsl version, entry point)
- [ ] pipelines
  - [ ] oneshot
  - [ ] synchronization based on dependency graph
  - [ ] manual / automatic dependency dependency graph
  - [ ] gpu buffers/images (?)
- [ ] highlevel interface
  - [ ] session
  - [ ] memory management (CPU / GPU, Buffer / Uniform / Image)
- [ ] package
  - [ ] pypi
  - [ ] python3

### Diary

* 30.01.2019
  * ran into memory alignment issues with uniforms (matrix of type 2x2, 2x3, 2x4)
  * less/no problems with buffers **and** std430 layout (not available for uniforms) 
  * thinking about dropping support for uniforms (or just specific types?)
  * when dropping uniforms the std430 layout could be enforced to be used everywhere (what about images?)

* 01.02.2019
  * found usable documentation and understood the alignment and offset situation after all
  * works smoothly for scalars and vectors
  * arrays are probably next

* 02.02.2019
  * when overwriting (assigning it a second time) the first value of a dynamic output array in a ssbo it resets the entire array to 0
  * arrays for basic types (uint, int, float, double) is working (also special case with numpy array)

* 02.02.2019 (2)
  * wrote basic SPIR-V decoder
  * booleans seem to be stored as uints
  * bytecode also contains the variable names (are optional by spec, but I guess the vulkan glsl compiler always includes them)
  * basically all information from the bindings is stored in the bytecode (and can be used to check user input)
  * only thing missing is the layout (std140, std430)
 