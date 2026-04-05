#pragma once
#include "../cuda/cuda_common.cuh"
#include <vector>
#include <cstdint>

struct SweepIntrinsicMeta {
    int sweep_number = -1;
    int64_t sweep_start_epoch_ms = 0;
    int64_t sweep_end_epoch_ms = 0;
    int64_t sweep_display_epoch_ms = 0;
    uint16_t radial_count = 0;
    uint16_t first_azimuth_number = 0;
    uint16_t last_azimuth_number = 0;
    uint32_t product_mask = 0;
    bool timing_exact = false;
    bool boundary_complete = false;
};

// Pre-computed GPU-ready sweep data (computed once at parse time)
struct PrecomputedSweep {
    SweepIntrinsicMeta meta;
    float elevation_angle = 0;
    int   num_radials = 0;
    std::vector<float> azimuths;
    std::vector<uint32_t> radial_time_offset_ms; // relative to sweep_start_epoch_ms
    struct ProductData {
        bool has_data = false;
        int  num_gates = 0;
        float first_gate_km = 0;
        float gate_spacing_km = 0;
        float scale = 0, offset = 0;
        std::vector<uint16_t> gates; // gate-major: [gate][radial]
    };
    ProductData products[NUM_PRODUCTS];
};
