#include "workstation_state.h"

#include <algorithm>

namespace workstation {

ShellState defaultShellState() {
    ShellState state;
    state.workspace = WorkspaceId::Live;
    state.context_tab = ContextDockTab::Inspect;
    state.time_mode = TimeDeckMode::LiveTail;
    state.pane_count = 1;
    state.context_dock_collapsed = false;
    state.assets_drawer_open = false;
    state.command_palette_open = false;
    state.panes[0].visible = true;
    return state;
}

RegionRects computeShellRects(int viewport_width, int viewport_height, bool dock_collapsed) {
    RegionRects rects;

    const int top_h = 76;
    const int time_h = 116;
    const int rail_w = 88;
    const int dock_w = dock_collapsed ? 0 : 336;

    rects.top_x = 0;
    rects.top_y = 0;
    rects.top_w = viewport_width;
    rects.top_h = top_h;

    rects.time_x = 0;
    rects.time_h = std::min(time_h, std::max(88, viewport_height / 5));
    rects.time_y = std::max(0, viewport_height - rects.time_h);
    rects.time_w = viewport_width;

    rects.rail_x = 0;
    rects.rail_y = top_h;
    rects.rail_w = rail_w;
    rects.rail_h = std::max(0, rects.time_y - top_h);

    rects.dock_w = dock_w;
    rects.dock_x = std::max(0, viewport_width - dock_w);
    rects.dock_y = top_h;
    rects.dock_h = std::max(0, rects.time_y - top_h);

    rects.canvas_x = rail_w;
    rects.canvas_y = top_h;
    rects.canvas_w = std::max(0, viewport_width - rail_w - dock_w);
    rects.canvas_h = std::max(0, rects.time_y - top_h);

    return rects;
}

} // namespace workstation
