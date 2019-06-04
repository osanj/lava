# Lava

* [Installation](#installation)
* [Support](#support)
* [Features](#features)
* [Example](#example)


## Installation

```
pip install lava
```

The Vulkan SDK needs to be installed and its environment variables need to be set.
See the LunarG guide for [Linux](https://vulkan.lunarg.com/doc/view/1.1.85.0/linux/getting_started.html) and [Windows](https://vulkan.lunarg.com/doc/view/1.1.85.0/windows/getting_started.html).

## Support

Sofar `lava` was tested on:
* Nvidia Tesla V100, Ubuntu 18.04 + Nvidia 415 (✓ works)
* Nvidia 1080 GTX, Ubuntu 18.04 + Nvidia 415 (✓ works)
* Nvidia 1080 GTX, Ubuntu 18.04 + Nvidia 396 (✗ does not work, driver issue)
* Nvidia 1080 GTX, Windows 10 (✓ works)
* Intel HD, Ubuntu 18.04 + Mesa (✓ works)


## Features

* automatic memory alignment and parsing of arbitrary complex block interfaces
  * supported: scalars (int, uint, float, double), vectors, matrices, multidimensional arrays and structs
  * partially supported: bool
  * not supported: dynamic arrays
  * further supported: ssbo's and ubo's, std140 and std430 layouts
  * multidimensional arrays of scalars, vectors or matrices are expected and parsed as respective numpy arrays
* cpu, gpu and staged buffers
* intuitive shader execution
  * block definitions are parsed from the compiled shader bytecode
  * buffers are bound with a single line of code
  * buffers and shaders are checked for compatibility, other sanity checks are performed


## Example

Below is the `lava` version of [Erkaman's vulkan minimal compute](https://github.com/Erkaman/vulkan_minimal_compute):

```python
import cv2 as cv
import lava as lv
import numpy as np

# start session on first device
session = lv.Session(lv.devices()[0])

# compile glsl code
shader_path = "shader.comp"
shader = lv.Shader(session, lv.compile_glsl(shader_path))

# allocate buffer and take its definition from the shader (see glsl: float[HEIGHT][WIDTH][3] imageData)
buffer_out = lv.StagedBuffer.from_shader(session, shader, binding=0)

# compute minimum amount of work groups
work_group_size = 32
x = int(3200 / float(work_group_size) + 0.5)
y = int(2400 / float(work_group_size) + 0.5)
z = 1

# bind buffer to binding 0 (see glsl: layout(std430, binding = 0) buffer buf)
stage = lv.Stage(shader, {0: buffer_out})
stage.record(x, y, z)
stage.run_and_wait()

# retrieve output as numpy array
im = buffer_out["imageData"]

cv.imwrite("mandelbrot.png", np.around(255 * im).astype(np.uint8))
```

```glsl
#version 450
#extension GL_ARB_separate_shader_objects : enable

#define WIDTH 3200
#define HEIGHT 2400
#define WORKGROUP_SIZE 32
layout ( local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;

layout(std430, binding = 0) buffer buf {
   float[HEIGHT][WIDTH][3] imageData;
};

void main() {
  if(gl_GlobalInvocationID.x >= WIDTH || gl_GlobalInvocationID.y >= HEIGHT)
    return;

  float x = float(gl_GlobalInvocationID.x) / float(WIDTH);
  float y = float(gl_GlobalInvocationID.y) / float(HEIGHT);

  vec2 uv = vec2(x, y);
  float n = 0.0;
  vec2 c = vec2(-.445, 0.0) + (uv - 0.5) * (2.0 + 1.7 * 0.2),
  z = vec2(0.0);
  const int M = 128;
  for (int i = 0; i < M; i++) {
    z = vec2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
    if (dot(z, z) > 2) break;
    n++;
  }
  
  float t = float(n) / float(M);
  vec3 d = vec3(0.3, 0.3 ,0.5);
  vec3 e = vec3(-0.2, -0.3 ,-0.5);
  vec3 f = vec3(2.1, 2.0, 3.0);
  vec3 g = vec3(0.0, 0.1, 0.0);
  vec3 color = d + e * cos( 6.28318 * (f * t + g) );

  int xIndex = int(gl_GlobalInvocationID.x);
  int yIndex = int(gl_GlobalInvocationID.y);
  imageData[yIndex][xIndex][0] = color.r;
  imageData[yIndex][xIndex][1] = color.g;
  imageData[yIndex][xIndex][2] = color.b;
}
```
