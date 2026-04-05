#include "workstation.h"

#include "app.h"
#include "nexrad/products.h"
#include "net/warnings.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ui {

namespace {

float stationDistanceKm(float lat1, float lon1, float lat2, float lon2) {
    constexpr float kDegToKm = 111.0f;
    const float meanLatRad = 0.5f * (lat1 + lat2) * 3.14159265f / 180.0f;
    const float dLatKm = (lat1 - lat2) * kDegToKm;
    const float dLonKm = (lon1 - lon2) * kDegToKm * std::cos(meanLatRad);
    return std::sqrt(dLatKm * dLatKm + dLonKm * dLonKm);
}

void computeAlertCentroid(const WarningPolygon& warning, float& outLat, float& outLon) {
    const size_t count = std::min(warning.lats.size(), warning.lons.size());
    if (count == 0) {
        outLat = 0.0f;
        outLon = 0.0f;
        return;
    }

    float latSum = 0.0f;
    float lonSum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        latSum += warning.lats[i];
        lonSum += warning.lons[i];
    }
    outLat = latSum / (float)count;
    outLon = lonSum / (float)count;
}

int findAlertIndexById(const std::vector<WarningPolygon>& warnings, const std::string& alertId) {
    if (alertId.empty())
        return -1;
    for (int i = 0; i < (int)warnings.size(); ++i) {
        if (warnings[i].id == alertId)
            return i;
    }
    return -1;
}

std::vector<int> candidateStationsForAlert(const WarningPolygon& warning,
                                           const std::vector<StationUiState>& stations) {
    float alertLat = 0.0f;
    float alertLon = 0.0f;
    computeAlertCentroid(warning, alertLat, alertLon);

    struct Candidate {
        int idx = -1;
        float distanceKm = 0.0f;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(stations.size());
    for (const auto& station : stations) {
        if (station.index < 0)
            continue;
        candidates.push_back({station.index, stationDistanceKm(alertLat, alertLon, station.lat, station.lon)});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.distanceKm < b.distanceKm; });

    std::vector<int> result;
    result.reserve(std::min<size_t>(4, candidates.size()));
    for (size_t i = 0; i < std::min<size_t>(4, candidates.size()); ++i)
        result.push_back(candidates[i].idx);
    return result;
}

WorkspaceTemplate recommendedTemplateForAlert(const WarningPolygon* warning) {
    if (!warning)
        return WorkspaceTemplate::None;
    if (warning->group == WarningGroup::Tornado)
        return WorkspaceTemplate::TornadoInterrogate;
    if (warning->group == WarningGroup::Severe)
        return WorkspaceTemplate::HailInterrogate;
    return WorkspaceTemplate::DualLinked;
}

void rebuildTransportBookmarks(ConsoleSession& session, const WarningPolygon* selectedWarning) {
    session.transport.bookmarks.clear();
    if (!selectedWarning)
        return;

    if (!selectedWarning->issue_time.empty())
        session.transport.bookmarks.push_back({"Issued", selectedWarning->issue_time});
    if (!selectedWarning->expire_time.empty())
        session.transport.bookmarks.push_back({"Expires", selectedWarning->expire_time});
}

void syncStationWorkflowFromApp(const App& app, ConsoleSession& session) {
    const int activeStation = app.activeStation();
    if (activeStation >= 0)
        session.stationWorkflow.focusedStationId = activeStation;

    if (app.autoTrackStation()) {
        session.stationWorkflow.followNearest = true;
        session.stationWorkflow.lockedStationId = -1;
        if (activeStation >= 0)
            session.stationWorkflow.hoveredStationId = activeStation;
    } else {
        session.stationWorkflow.followNearest = false;
        session.stationWorkflow.lockedStationId = activeStation;
    }
}

void syncPaneSelectionsFromApp(const App& app, ConsoleSession& session) {
    const int activeStation = app.activeStation();
    const int activeTilt = app.activeTilt();
    const float activeTiltAngle = app.activeTiltAngle();

    for (int i = 0; i < 4; ++i) {
        PaneState& pane = session.panes[i];
        pane.selection.product = (i == 0) ? app.activeProduct() : app.radarPanelProduct(i);
        pane.selection.tilt.sweepIndex = (i == 0) ? activeTilt : app.radarPanelTilt(i);
        pane.selection.tilt.elevationDeg =
            (pane.selection.tilt.sweepIndex == activeTilt)
                ? activeTiltAngle
                : (float)pane.selection.tilt.sweepIndex;

        if (pane.selection.stationId < 0 || pane.links.station == session.panes[0].links.station)
            pane.selection.stationId = activeStation;
    }

    session.panes[0].kind = app.mode3D() ? PaneKind::Volume3D :
                            app.crossSection() ? PaneKind::CrossSection :
                                                 PaneKind::Radar2D;
    for (int i = 1; i < 4; ++i)
        session.panes[i].kind = PaneKind::Radar2D;
}

void syncTransportFromApp(const App& app, ConsoleSession& session) {
    const TransportSnapshot snapshot = app.transportSnapshot();
    session.transport.stream = snapshot.archive_mode ? StreamKind::Archive :
                               snapshot.snapshot_mode ? StreamKind::Snapshot :
                                                        StreamKind::Live;
    session.transport.mode = snapshot.archive_mode ? TimeDeckMode::Archive :
                             snapshot.snapshot_mode ? TimeDeckMode::Snapshot :
                             snapshot.review_enabled ? TimeDeckMode::Review :
                                                       TimeDeckMode::LiveTail;
    session.transport.isAdvancing = snapshot.playing;
    if (session.transport.isAdvancing)
        session.transport.playIntent = true;
    if (!snapshot.review_enabled && !snapshot.buffering && !snapshot.archive_mode)
        session.transport.playIntent = false;
    session.transport.requestedFrames = snapshot.requested_frames;
    session.transport.readyFrames = snapshot.ready_frames;
    session.transport.loadingFrames = snapshot.loading_frames;
    session.transport.cursorFrame = snapshot.cursor_frame;
    session.transport.rateFps = snapshot.rate_fps;
    session.transport.currentLabel = snapshot.current_label;
}

void syncWorkspaceFromApp(const App& app, ConsoleSession& session) {
    if (!session.initialized) {
        if (app.mode3D() || app.crossSection())
            session.activeWorkspace = WorkspaceId::Volume;
        else if (app.m_historicMode || app.snapshotMode())
            session.activeWorkspace = WorkspaceId::Archive;
        else
            session.activeWorkspace = app.radarPanelLayout() == RadarPanelLayout::Single
                ? WorkspaceId::Live
                : WorkspaceId::Compare;
        return;
    }

    if (app.mode3D() || app.crossSection()) {
        session.activeWorkspace = WorkspaceId::Volume;
        return;
    }
    if (app.m_historicMode || app.snapshotMode()) {
        session.activeWorkspace = WorkspaceId::Archive;
        return;
    }
    if (session.activeWorkspace == WorkspaceId::Volume || session.activeWorkspace == WorkspaceId::Archive) {
        session.activeWorkspace = app.radarPanelLayout() == RadarPanelLayout::Single
            ? WorkspaceId::Live
            : WorkspaceId::Compare;
    }
}

void syncAlertFocus(const std::vector<StationUiState>& stations,
                    const std::vector<WarningPolygon>& warnings,
                    ConsoleSession& session,
                    const App& app) {
    session.alertFocus.selectedAlertId = app.selectedAlert().selected_alert_id;
    session.alertFocus.candidateStations = app.selectedAlert().candidate_stations;
    const int alertIdx = findAlertIndexById(warnings, session.alertFocus.selectedAlertId);
    const WarningPolygon* selectedWarning = (alertIdx >= 0) ? &warnings[alertIdx] : nullptr;
    if (selectedWarning) {
        if (session.alertFocus.candidateStations.empty())
            session.alertFocus.candidateStations = candidateStationsForAlert(*selectedWarning, stations);
        session.alertFocus.recommendedWorkspace = WorkspaceId::Warning;
        session.alertFocus.recommendedTemplate = recommendedTemplateForAlert(selectedWarning);
    } else {
        session.alertFocus.candidateStations.clear();
        session.alertFocus.recommendedTemplate = WorkspaceTemplate::None;
    }
    rebuildTransportBookmarks(session, selectedWarning);
}

void propagateStationSelection(ConsoleSession& session, int sourcePane, int stationId) {
    const int group = session.panes[sourcePane].links.station;
    for (auto& pane : session.panes) {
        if (pane.links.station == group)
            pane.selection.stationId = stationId;
    }
}

void propagateTiltSelection(ConsoleSession& session, int sourcePane) {
    const int group = session.panes[sourcePane].links.tilt;
    const TiltSelection sourceTilt = session.panes[sourcePane].selection.tilt;
    for (auto& pane : session.panes) {
        if (pane.links.tilt == group)
            pane.selection.tilt = sourceTilt;
    }
}

} // namespace

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
        case ContextDockTab::Archive: return "Archive";
        case ContextDockTab::Tools: return "Tools";
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

const char* paneRoleLabel(PaneRole role) {
    switch (role) {
        case PaneRole::Primary: return "Primary";
        case PaneRole::Reflectivity: return "Reflectivity";
        case PaneRole::Velocity: return "Velocity";
        case PaneRole::CorrelationCoeff: return "Correlation";
        case PaneRole::HigherTiltTrend: return "Higher Tilt";
        case PaneRole::Comparison: return "Comparison";
        case PaneRole::ArchiveReference: return "Archive Ref";
        case PaneRole::Custom: return "Custom";
        default: return "Pane";
    }
}

ConsoleSession defaultConsoleSession() {
    ConsoleSession session;
    std::strncpy(session.stationWorkflow.searchQuery, "KTLX", sizeof(session.stationWorkflow.searchQuery));
    session.transport.requestedFrames = 8;
    session.transport.rateFps = 5.0f;

    for (int i = 0; i < 4; ++i) {
        session.panes[i].paneId = i;
        session.panes[i].selection.stationId = -1;
        session.panes[i].selection.product = (i == 0) ? PROD_REF :
                                             (i == 1) ? PROD_VEL :
                                             (i == 2) ? PROD_CC :
                                                        PROD_REF;
        session.panes[i].selection.tilt.sweepIndex = 0;
        session.panes[i].selection.tilt.elevationDeg = 0.5f;
        session.panes[i].links.geo = 0;
        session.panes[i].links.time = 0;
        session.panes[i].links.station = 0;
        session.panes[i].links.tilt = (i == 3) ? 3 : 0;
    }

    session.panes[0].role = PaneRole::Primary;
    session.panes[1].role = PaneRole::Velocity;
    session.panes[2].role = PaneRole::CorrelationCoeff;
    session.panes[3].role = PaneRole::HigherTiltTrend;
    return session;
}

void syncConsoleSessionFromApp(const App& app,
                               const std::vector<StationUiState>& stations,
                               const std::vector<WarningPolygon>& warnings,
                               ConsoleSession& session) {
    syncStationWorkflowFromApp(app, session);
    syncPaneSelectionsFromApp(app, session);
    syncTransportFromApp(app, session);
    syncWorkspaceFromApp(app, session);
    syncAlertFocus(stations, warnings, session, app);
    session.initialized = true;
}

void applyWorkspaceToApp(App& app, ConsoleSession& session) {
    if (session.activeWorkspace != WorkspaceId::Volume) {
        if (app.mode3D())
            app.toggle3D();
        if (app.crossSection())
            app.toggleCrossSection();
    }

    switch (session.activeWorkspace) {
        case WorkspaceId::Live:
            app.setRadarPanelLayout(RadarPanelLayout::Single);
            break;
        case WorkspaceId::Compare:
            app.setRadarPanelLayout(RadarPanelLayout::Dual);
            break;
        case WorkspaceId::Warning:
            app.setRadarPanelLayout(RadarPanelLayout::Quad);
            break;
        case WorkspaceId::Volume:
        case WorkspaceId::Archive:
        case WorkspaceId::Tools:
        case WorkspaceId::Assets:
            break;
    }
}

void setWorkspace(App& app, ConsoleSession& session, WorkspaceId workspace) {
    session.activeWorkspace = workspace;
    applyWorkspaceToApp(app, session);
}

void applySelectedPaneToApp(App& app, ConsoleSession& session) {
    const int paneIdx = std::clamp(session.activePaneIndex, 0, 3);
    PaneState& pane = session.panes[paneIdx];
    if (pane.selection.stationId >= 0 && pane.selection.stationId != app.activeStation())
        focusStation(app, session, pane.selection.stationId, false, -1.0);
    setPaneProduct(app, session, paneIdx, pane.selection.product);
    setPaneTilt(app, session, paneIdx, pane.selection.tilt.sweepIndex);
}

void updateHoveredStation(ConsoleSession& session, int stationId) {
    session.stationWorkflow.hoveredStationId = stationId;
}

void focusStation(App& app, ConsoleSession& session, int stationId, bool centerView, double zoom) {
    if (stationId < 0)
        return;

    session.stationWorkflow.focusedStationId = stationId;
    if (!session.stationWorkflow.followNearest)
        session.stationWorkflow.lockedStationId = stationId;
    propagateStationSelection(session, session.activePaneIndex, stationId);
    app.selectStation(stationId, centerView, zoom);
    app.setAutoTrackStation(session.stationWorkflow.followNearest);
}

void setPaneProduct(App& app, ConsoleSession& session, int paneIndex, int product) {
    if (paneIndex < 0 || paneIndex >= (int)session.panes.size())
        return;
    session.panes[paneIndex].selection.product = product;
    if (paneIndex == 0)
        app.setProduct(product);
    else
        app.setRadarPanelProduct(paneIndex, product);
}

void setPaneTilt(App& app, ConsoleSession& session, int paneIndex, int tiltIndex) {
    if (paneIndex < 0 || paneIndex >= (int)session.panes.size())
        return;

    tiltIndex = std::max(0, std::min(tiltIndex, std::max(0, app.maxTilts() - 1)));
    session.panes[paneIndex].selection.tilt.sweepIndex = tiltIndex;
    session.panes[paneIndex].selection.tilt.elevationDeg =
        (tiltIndex == app.activeTilt()) ? app.activeTiltAngle() : (float)tiltIndex;
    propagateTiltSelection(session, paneIndex);

    for (int i = 0; i < (int)session.panes.size(); ++i) {
        if (session.panes[i].selection.tilt.sweepIndex < 0)
            continue;
        if (i == 0 && session.panes[i].links.tilt == session.panes[paneIndex].links.tilt) {
            app.setTilt(session.panes[i].selection.tilt.sweepIndex);
            session.panes[i].selection.tilt.elevationDeg = app.activeTiltAngle();
        } else if (i > 0 && session.panes[i].links.tilt == session.panes[paneIndex].links.tilt) {
            app.setRadarPanelTilt(i, session.panes[i].selection.tilt.sweepIndex);
        }
    }
}

void toggleTransportPlay(App& app, ConsoleSession& session) {
    const bool desired = !session.transport.playIntent;
    session.transport.playIntent = desired;

    if (session.transport.stream == StreamKind::Archive) {
        app.transportSetPlay(desired);
        return;
    }

    if (session.transport.stream != StreamKind::Live)
        return;

    app.transportSetPlay(desired);
}

void transportJumpLive(App& app, ConsoleSession& session) {
    app.transportJumpLive();
    session.transport.cursorFrame = app.transportSnapshot().cursor_frame;
}

void transportSetLoopEnabled(App& app, ConsoleSession& session, bool enabled) {
    if (session.transport.stream != StreamKind::Live)
        return;
    app.transportSetReviewEnabled(enabled);
    session.transport.mode = enabled ? TimeDeckMode::Review : TimeDeckMode::LiveTail;
    if (!enabled)
        session.transport.playIntent = false;
}

void transportSetRequestedFrames(App& app, ConsoleSession& session, int frames) {
    if (session.transport.stream != StreamKind::Live)
        return;
    app.transportSetRequestedFrames(frames);
    session.transport.requestedFrames = app.transportSnapshot().requested_frames;
}

void transportSetRate(App& app, ConsoleSession& session, float fps) {
    if (session.transport.stream == StreamKind::Archive)
        return;
    app.transportSetRate(fps);
    session.transport.rateFps = app.transportSnapshot().rate_fps;
}

void transportSeekFrame(App& app, ConsoleSession& session, int frameIndex) {
    if (session.transport.stream == StreamKind::Snapshot)
        return;
    app.transportSeekFrame(frameIndex);
    session.transport.cursorFrame = app.transportSnapshot().cursor_frame;
    session.transport.playIntent = false;
}

void transportStep(App& app, ConsoleSession& session, int delta) {
    if (session.transport.stream == StreamKind::Snapshot)
        return;
    const TransportSnapshot snapshot = app.transportSnapshot();
    if (snapshot.total_frames <= 0)
        return;
    const int next = std::max(0, std::min(snapshot.cursor_frame + delta, std::max(0, snapshot.total_frames - 1)));
    app.transportSeekFrame(next);
    session.transport.cursorFrame = next;
    session.transport.playIntent = false;
}

void selectAlert(App& app,
                 ConsoleSession& session,
                 const std::vector<WarningPolygon>& warnings,
                 const std::vector<StationUiState>& stations,
                 const std::string& alertId) {
    session.alertFocus.selectedAlertId = alertId;
    const int idx = findAlertIndexById(warnings, alertId);
    const WarningPolygon* selected = (idx >= 0) ? &warnings[idx] : nullptr;
    if (selected) {
        session.activeWorkspace = WorkspaceId::Warning;
        session.activeDockTab = ContextDockTab::Alerts;
        session.alertFocus.candidateStations = candidateStationsForAlert(*selected, stations);
        session.alertFocus.recommendedWorkspace = WorkspaceId::Warning;
        session.alertFocus.recommendedTemplate = recommendedTemplateForAlert(selected);
        app.setSelectedAlert(alertId, session.alertFocus.candidateStations,
                             session.alertFocus.candidateStations.empty() ? -1 : session.alertFocus.candidateStations.front());
    } else {
        session.alertFocus.candidateStations.clear();
        session.alertFocus.recommendedTemplate = WorkspaceTemplate::None;
        app.clearSelectedAlert();
    }
    rebuildTransportBookmarks(session, selected);
}

void activateTornadoInterrogate(App& app,
                                ConsoleSession& session,
                                const std::vector<WarningPolygon>& warnings,
                                const std::vector<StationUiState>& stations) {
    const int idx = findAlertIndexById(warnings, session.alertFocus.selectedAlertId);
    if (idx >= 0)
        session.alertFocus.candidateStations = candidateStationsForAlert(warnings[idx], stations);

    setWorkspace(app, session, WorkspaceId::Warning);
    session.activeDockTab = ContextDockTab::Alerts;
    session.activePaneIndex = 0;

    for (auto& pane : session.panes) {
        pane.kind = PaneKind::Radar2D;
        pane.links.geo = 0;
        pane.links.time = 0;
        pane.links.station = 0;
        pane.links.tilt = 0;
    }

    session.panes[0].role = PaneRole::Reflectivity;
    session.panes[1].role = PaneRole::Velocity;
    session.panes[2].role = PaneRole::CorrelationCoeff;
    session.panes[3].role = PaneRole::HigherTiltTrend;
    session.panes[3].links.tilt = 3;

    setPaneProduct(app, session, 0, PROD_REF);
    setPaneProduct(app, session, 1, PROD_VEL);
    setPaneProduct(app, session, 2, PROD_CC);
    setPaneProduct(app, session, 3, PROD_VEL);
    setPaneTilt(app, session, 0, 0);
    setPaneTilt(app, session, 3, std::min(app.activeTilt() + 1, std::max(0, app.maxTilts() - 1)));

    if (!session.alertFocus.candidateStations.empty()) {
        session.stationWorkflow.followNearest = false;
        session.stationWorkflow.lockedStationId = session.alertFocus.candidateStations.front();
        focusStation(app, session, session.alertFocus.candidateStations.front(), true, 140.0);
        app.setAutoTrackStation(false);
    }
}

} // namespace ui
