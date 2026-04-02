#pragma once

#include <array>

namespace workstation {

enum class WorkspaceId {
    Live = 0,
    Compare,
    Archive,
    Warning,
    Volume,
    Tools,
    Assets
};

enum class ContextDockTab {
    Inspect = 0,
    Alerts,
    Layers,
    Assets,
    Session
};

enum class TimeDeckMode {
    LiveTail = 0,
    Review,
    Archive,
    Snapshot
};

struct PaneLinks {
    bool geo = true;
    bool time = true;
    bool station = true;
    bool tilt = true;
};

struct PaneState {
    bool visible = false;
    int station_index = -1;
    int product = 0;
    int tilt = 0;
    PaneLinks links = {};
};

struct ShellState {
    WorkspaceId workspace = WorkspaceId::Live;
    ContextDockTab context_tab = ContextDockTab::Inspect;
    TimeDeckMode time_mode = TimeDeckMode::LiveTail;
    int pane_count = 1;
    bool assets_drawer_open = false;
    bool context_dock_collapsed = false;
    bool command_palette_open = false;
    std::array<PaneState, 4> panes = {};
};

struct RegionRects {
    int top_x = 0;
    int top_y = 0;
    int top_w = 0;
    int top_h = 0;

    int rail_x = 0;
    int rail_y = 0;
    int rail_w = 0;
    int rail_h = 0;

    int canvas_x = 0;
    int canvas_y = 0;
    int canvas_w = 0;
    int canvas_h = 0;

    int dock_x = 0;
    int dock_y = 0;
    int dock_w = 0;
    int dock_h = 0;

    int time_x = 0;
    int time_y = 0;
    int time_w = 0;
    int time_h = 0;
};

ShellState defaultShellState();
RegionRects computeShellRects(int viewport_width, int viewport_height, bool dock_collapsed);

} // namespace workstation
