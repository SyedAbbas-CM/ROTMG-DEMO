// server/wasm/BulletUpdate.cpp
#include <emscripten.h>
#include <stdint.h>

/**
 * updateBullets:
 *   - x,y,vx,vy,life are float arrays
 *   - bulletCountPtr points to an int storing the number of active bullets
 *   - dt is a float for deltaTime
 *
 * Moves bullets by vx*dt, vy*dt, decrements life, swap-removes if expired.
 */
extern "C" {
EMSCRIPTEN_KEEPALIVE
void updateBullets(
  float* x, float* y,
  float* vx, float* vy,
  float* life,
  int32_t* bulletCountPtr,
  float dt
) {
  int count = *bulletCountPtr;
  for (int i = 0; i < count; i++) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    life[i] -= dt;

    // If bullet is expired, swap-remove
    if (life[i] <= 0.f) {
      int last = count - 1;
      if (i != last) {
        x[i]    = x[last];
        y[i]    = y[last];
        vx[i]   = vx[last];
        vy[i]   = vy[last];
        life[i] = life[last];
      }
      count--;
      i--;
    }
  }
  *bulletCountPtr = count;
}
}
