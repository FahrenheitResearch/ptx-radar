#include "workstation.h"
#include "app.h"

namespace ui {

const char* workspaceLabel(WorkspaceId workspace) {
    switch (workspace) {
        case WorkspaceId::Live: return "Live";
        case WorkspaceId::Compare: return "Compare";
        case WorkspaceId::Archive: return "Archive";
        case WorkspaceId::Warning: return "Warning";
        case WorkspaceId::Volume: return "Volume / XS";
        case WorkspaceId::Tools: return "Tools";
        case WorkspaceId::Assets: return "Assets";
        default: return "Workspace";
    }
}

const char* contextDockTabLabel(ContextDockTab tab) {
    switch (tab) {
        case ContextDockTab::Inspect: return "Inspect";
        case ContextDockTab::Alerts: return "Alerts";
        case ContextDockTab::Layers: return "Layers";
        case ContextDockTab::Assets: return "Assets";
        case ContextDockTab::Session: return "Session";
        default: return "Dock";
    }
}

const char* timeDeckModeLabel(TimeDeckMode mode) {
    switch (mode) {
        case TimeDeckMode::LiveTail: return "Live Tail";
        case TimeDeckMode::Review: return "Review";
        case TimeDeckMode::Archive: return "Archive";
        case TimeDeckMode::Snapshot: return "Snapshot";
        default: return "Time";
    }
}

void syncWorkstationStateFromApp(const App& app, WorkstationState& state) {
    if (app.mode3D() || app.crossSection()) {
        state.activeWorkspace = WorkspaceId::Volume;
    } else if (app.m_historicMode || app.snapshotMode()) {
        state.activeWorkspace = WorkspaceId::Archive;
    } else if (app.radarPanelLayout() != RadarPanelLayout::Single) {
        state.activeWorkspace = WorkspaceId::Compare;
    } else {
        state.activeWorkspace = WorkspaceId::Live;
    }

    if (app.m_historicMode) {
        state.timeMode = TimeDeckMode::Archive;
    } else if (app.snapshotMode()) {
        state.timeMode = TimeDeckMode::Snapshot;
    } else if (app.liveLoopEnabled() && (app.liveLoopViewingHistory() || app.liveLoopAvailableFrames() > 0)) {
        state.timeMode = TimeDeckMode::Review;
    } else {
        state.timeMode = TimeDeckMode::LiveTail;
    }
}

} // namespace ui
