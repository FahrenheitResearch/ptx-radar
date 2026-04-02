#pragma once

class App;

namespace ui {

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

struct PaneLinkState {
    bool geo = true;
    bool time = true;
    bool station = true;
    bool tilt = true;
};

struct WorkstationState {
    WorkspaceId activeWorkspace = WorkspaceId::Live;
    ContextDockTab activeDockTab = ContextDockTab::Inspect;
    TimeDeckMode timeMode = TimeDeckMode::LiveTail;
    bool contextDockOpen = true;
    bool workspaceRailExpanded = false;
    PaneLinkState paneLinks[4] = {};
};

const char* workspaceLabel(WorkspaceId workspace);
const char* contextDockTabLabel(ContextDockTab tab);
const char* timeDeckModeLabel(TimeDeckMode mode);
void syncWorkstationStateFromApp(const App& app, WorkstationState& state);

} // namespace ui
