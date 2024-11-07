#include "raylib.h"
#include "screens.h"
#include <vector>
#include <raymath.h>

#define IX(x, y) ((x) + (y) * N)

struct FluidCube {
    int size;
    float dt;
    float diff;
    float visc;

    std::vector<float> s;
    std::vector<float> density;
    std::vector<float> Vx;
    std::vector<float> Vy;
    std::vector<float> Vx0;
    std::vector<float> Vy0;
};

int SCALE = 10;
int N = 64;
FluidCube cube;
Vector2 mousePosition;
Vector2 preMousePosition = { 0, 0 };
static int framesCounter = 0;
static int finishScreen = 0;
Camera camera = { 0 };

void FluidCubeCreate(FluidCube& cube, int diffusion, int viscosity, float dt) {
    cube.size = N;
    cube.dt = dt;
    cube.diff = diffusion;
    cube.visc = viscosity;

    cube.s.resize(N * N, 0.0f);
    cube.density.resize(N * N, 0.0f);
    cube.Vx.resize(N * N, 0.0f);
    cube.Vy.resize(N * N, 0.0f);
    cube.Vx0.resize(N * N, 0.0f);
    cube.Vy0.resize(N * N, 0.0f);
}

void FluidCubeAddDensity(FluidCube& cube, int x, int y, float amount) {
    if (x >= 0 && x < N && y >= 0 && y < N) {
        cube.density[IX(x, y)] += amount;
    }
}

void FluidCubeAddVelocity(FluidCube& cube, int x, int y, float amountX, float amountY) {
    if (x >= 0 && x < N && y >= 0 && y < N) {
        int index = IX(x, y);
        cube.Vx[index] += amountX;
        cube.Vy[index] += amountY;
    }
}

static void set_bnd(int b, std::vector<float>& x) {
    for (int i = 1; i < N - 1; i++) {
        if (IX(i, 0) < x.size() && IX(i, 1) < x.size() && IX(i, N - 1) < x.size() && IX(i, N - 2) < x.size()) {
            x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
            x[IX(i, N - 1)] = b == 2 ? -x[IX(i, N - 2)] : x[IX(i, N - 2)];
        }
    }
    for (int j = 1; j < N - 1; j++) {
        if (IX(0, j) < x.size() && IX(1, j) < x.size() && IX(N - 1, j) < x.size() && IX(N - 2, j) < x.size()) {
            x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
            x[IX(N - 1, j)] = b == 1 ? -x[IX(N - 2, j)] : x[IX(N - 2, j)];
        }
    }
    if (IX(0, 0) < x.size() && IX(1, 0) < x.size() && IX(0, 1) < x.size()) {
        x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    }
    if (IX(0, N - 1) < x.size() && IX(1, N - 1) < x.size() && IX(0, N - 2) < x.size()) {
        x[IX(0, N - 1)] = 0.5f * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
    }
    if (IX(N - 1, 0) < x.size() && IX(N - 2, 0) < x.size() && IX(N - 1, 1) < x.size()) {
        x[IX(N - 1, 0)] = 0.5f * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
    }
    if (IX(N - 1, N - 1) < x.size() && IX(N - 2, N - 1) < x.size() && IX(N - 1, N - 2) < x.size()) {
        x[IX(N - 1, N - 1)] = 0.5f * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);
    }
}

static void lin_solve(int b, std::vector<float>& x, std::vector<float>& x0, float a, float c, int iter) {
    float cRecip = 1.0 / c;
    for (int k = 0; k < iter; k++) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                if (IX(i, j) < x.size() && IX(i + 1, j) < x.size() && IX(i - 1, j) < x.size() && IX(i, j + 1) < x.size() && IX(i, j - 1) < x.size()) {
                    x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i + 1, j)] + x[IX(i - 1, j)] + x[IX(i, j + 1)] + x[IX(i, j - 1)])) * cRecip;
                }
            }
        }
        set_bnd(b, x);
    }
}

static void diffuse(int b, std::vector<float>& x, std::vector<float>& x0, float diff, float dt, int iter) {
    float a = dt * diff * (N - 2) * (N - 2);
    lin_solve(b, x, x0, a, 1 + 6 * a, iter);
}

static void project(std::vector<float>& velocX, std::vector<float>& velocY, std::vector<float>& p, std::vector<float>& div, int iter) {
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            if (IX(i, j) < velocX.size() && IX(i + 1, j) < velocX.size() && IX(i - 1, j) < velocX.size() && IX(i, j + 1) < velocY.size() && IX(i, j - 1) < velocY.size()) {
                div[IX(i, j)] = -0.5f * (velocX[IX(i + 1, j)] - velocX[IX(i - 1, j)] + velocY[IX(i, j + 1)] - velocY[IX(i, j - 1)]) / N;
                p[IX(i, j)] = 0;
            }
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);
    lin_solve(0, p, div, 1, 6, iter);
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            if (IX(i, j) < velocX.size() && IX(i + 1, j) < p.size() && IX(i - 1, j) < p.size() && IX(i, j + 1) < p.size() && IX(i, j - 1) < p.size()) {
                velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
                velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
            }
        }
    }
    set_bnd(1, velocX);
    set_bnd(2, velocY);
}

static void advect(int b, std::vector<float>& d, std::vector<float>& d0, std::vector<float>& velocX, std::vector<float>& velocY, float dt) {
    float dtx = dt * (N - 2);
    float dty = dt * (N - 2);
    float Nfloat = N;
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            if (IX(i, j) < d.size() && IX(i, j) < velocX.size() && IX(i, j) < velocY.size()) {
                float x = i - dtx * velocX[IX(i, j)];
                float y = j - dty * velocY[IX(i, j)];
                if (x < 0.5f) x = 0.5f;
                if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
                if (y < 0.5f) y = 0.5f;
                if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
                int i0 = (int)x;
                int i1 = i0 + 1;
                int j0 = (int)y;
                int j1 = j0 + 1;
                float s1 = x - i0;
                float s0 = 1.0f - s1;
                float t1 = y - j0;
                float t0 = 1.0f - t1;

                if (IX(i, j) < d.size() && IX(i0, j0) < d0.size() && IX(i0, j1) < d0.size() && IX(i1, j0) < d0.size() && IX(i1, j1) < d0.size()) {
                    d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
                }
            }
        }
    }
    set_bnd(b, d);
}

void FluidCubeStep(FluidCube& cube) {
    int iter = 15;
    diffuse(1, cube.Vx0, cube.Vx, cube.visc, cube.dt, iter);
    diffuse(2, cube.Vy0, cube.Vy, cube.visc, cube.dt, iter);
    project(cube.Vx0, cube.Vy0, cube.Vx, cube.Vy, iter);
    advect(1, cube.Vx, cube.Vx0, cube.Vx0, cube.Vy0, cube.dt);
    advect(2, cube.Vy, cube.Vy0, cube.Vx0, cube.Vy0, cube.dt);
    project(cube.Vx, cube.Vy, cube.Vx0, cube.Vy0, iter);
    diffuse(0, cube.s, cube.density, cube.diff, cube.dt, iter);
    advect(0, cube.density, cube.s, cube.Vx, cube.Vy, cube.dt);
}

void InitGameplayScreen(void) {
    framesCounter = 0;
    finishScreen = 0;
    FluidCubeCreate(cube, 0, 0, 0.1f);
}

void UpdateGameplayScreen(void) {
    mousePosition = GetMousePosition();
    int x = mousePosition.x / SCALE;
    int y = mousePosition.y / SCALE;
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && x >= 0 && x < N && y >= 0 && y < N) {
        FluidCubeAddDensity(cube, x, y, 100);
        float amtx = preMousePosition.x - mousePosition.x;
        float amty = preMousePosition.y - mousePosition.y;
        FluidCubeAddVelocity(cube, x, y, amtx / SCALE, amty / SCALE);
    }
    preMousePosition = mousePosition;
}

void DrawGameplayScreen(void) {
    ClearBackground(RAYWHITE);
    FluidCubeStep(cube);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (IX(x, y) < cube.density.size()) {
                float d = cube.density[IX(x, y)];
                unsigned char colorValue = (unsigned char)Clamp(d, 0, 255);
                Color color = { colorValue, colorValue, colorValue, 255 };
                DrawRectangle(x * SCALE, y * SCALE, SCALE, SCALE, color);
            }
        }
    }
    DrawFPS(10, 10);
}

void UnloadGameplayScreen(void) {
    // TODO: Unload GAMEPLAY screen variables here!
}

int FinishGameplayScreen(void) {
    return finishScreen;
}
