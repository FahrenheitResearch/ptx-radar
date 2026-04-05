#pragma once

#include <array>
#include <string>
#include <vector>

class App;
struct StationUiState;
struct WarningPolygon;

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
    Archive,
    Tools,
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

enum class StreamKind {
    Live = 0,
    Archive,
    Snapshot
};

enum class CursorMode {
    LiveTail = 0,
    RelativeFrame,
    AbsoluteFrame
};

enum class PaneKind {
    Radar2D = 0,
    CrossSection,
    Volume3D
};

enum class PaneRole {
    Primary = 0,
    Reflectivity,
    Velocity,
    CorrelationCoeff,
    HigherTiltTrend,
    Comparison,
    ArchiveReference,
    Custom
};

enum class WorkspaceTemplate {
    None = 0,
    SoloLive,
    DualLinked,
    TornadoInterrogate,
    HailInterrogate,
    ArchiveReview,
    VolumeInspect,
    CrossSection
};

struct PaneLinkGroups {
    int geo = 0;
    int time = 0;
    int station = 0;
    int tilt = 0;
};

struct StationWorkflowState {
    int hoveredStationId = -1;
    int focusedStationId = -1;
    int lockedStationId = -1;
    bool followNearest = true;
    char searchQuery[16] = "KTLX";
};

struct TiltSelection {
    int sweepIndex = 0;
    float elevationDeg = 0.5f;
};

struct TimeBinding {
    bool followsGlobalTransport = true;
    int relativeFrameOffset = 0;
};

struct SelectionState {
    int stationId = -1;
    int product = 0;
    TiltSelection tilt;
    TimeBinding time;
};

struct ProbeState {
    bool pinned = false;
    float lat = 0.0f;
    float lon = 0.0f;
};

struct PaneState {
    int paneId = 0;
    PaneKind kind = PaneKind::Radar2D;
    PaneRole role = PaneRole::Primary;
    SelectionState selection;
    PaneLinkGroups links;
    ProbeState probe;
};

struct Bookmark {
    std::string label;
    std::string timeText;
};

struct TransportState {
    StreamKind stream = StreamKind::Live;
    TimeDeckMode mode = TimeDeckMode::LiveTail;
    bool playIntent = false;
    bool isAdvancing = false;
    CursorMode cursorMode = CursorMode::LiveTail;
    float rateFps = 5.0f;
    int requestedFrames = 8;
    int readyFrames = 0;
    int loadingFrames = 0;
    int cursorFrame = 0;
    std::string currentLabel;
    std::vector<Bookmark> bookmarks;
};

struct AlertFocusState {
    std::string hoveredAlertId;
    std::string selectedAlertId;
    std::vector<int> candidateStations;
    WorkspaceId recommendedWorkspace = WorkspaceId::Warning;
    WorkspaceTemplate recommendedTemplate = WorkspaceTemplate::None;
};

struct ConsoleSession {
    bool initialized = false;
    WorkspaceId activeWorkspace = WorkspaceId::Live;
    ContextDockTab activeDockTab = ContextDockTab::Inspect;
    bool contextDockOpen = true;
    bool workspaceRailExpanded = false;
    float workspaceRailWidth = 100.0f;
    int activePaneIndex = 0;
    StationWorkflowState stationWorkflow;
    TransportState transport;
    AlertFocusState alertFocus;
    std::array<PaneState, 4> panes;
};

const char* workspaceLabel(WorkspaceId workspace);
const char* contextDockTabLabel(ContextDockTab tab);
const char* timeDeckModeLabel(TimeDeckMode mode);
const char* paneRoleLabel(PaneRole role);

ConsoleSession defaultConsoleSession();
void syncConsoleSessionFromApp(const App& app,
                               const std::vector<StationUiState>& stations,
                               const std::vector<WarningPolygon>& warnings,
                               ConsoleSession& session);
void applyWorkspaceToApp(App& app, ConsoleSession& session);
void applySelectedPaneToApp(App& app, ConsoleSession& session);
void setWorkspace(App& app, ConsoleSession& session, WorkspaceId workspace);
void updateHoveredStation(ConsoleSession& session, int stationId);
void focusStation(App& app, ConsoleSession& session, int stationId,
                  bool centerView = false, double zoom = -1.0);
void setPaneProduct(App& app, ConsoleSession& session, int paneIndex, int product);
void setPaneTilt(App& app, ConsoleSession& session, int paneIndex, int tiltIndex);
void toggleTransportPlay(App& app, ConsoleSession& session);
void transportJumpLive(App& app, ConsoleSession& session);
void transportSetLoopEnabled(App& app, ConsoleSession& session, bool enabled);
void transportSetRequestedFrames(App& app, ConsoleSession& session, int frames);
void transportSetRate(App& app, ConsoleSession& session, float fps);
void transportSeekFrame(App& app, ConsoleSession& session, int frameIndex);
void transportStep(App& app, ConsoleSession& session, int delta);
void selectAlert(App& app,
                 ConsoleSession& session,
                 const std::vector<WarningPolygon>& warnings,
                 const std::vector<StationUiState>& stations,
                 const std::string& alertId);
void activateTornadoInterrogate(App& app,
                                ConsoleSession& session,
                                const std::vector<WarningPolygon>& warnings,
                                const std::vector<StationUiState>& stations);

} // namespace ui
