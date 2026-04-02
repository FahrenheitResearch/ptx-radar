#include "ui.h"

#include "app.h"
#include "historic.h"
#include "net/warnings.h"
#include "nexrad/products.h"
#include "nexrad/stations.h"
#include "workstation_state.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace ui {

namespace {

bool g_uiWantsMouseCapture = false;
WorkstationShellState g_shellState = {};

constexpr float kTopBarHeight = 72.0f;
constexpr float kRailWidth = 86.0f;
constexpr float kDockWidth = 360.0f;
constexpr float kTimeDeckHeight = 102.0f;

const char* workspaceLabel(WorkspaceId workspace) {
    switch (workspace) {
        case WorkspaceId::Live: return "Live";
        case WorkspaceId::Compare: return "Compare";
        case WorkspaceId::Archive: return "Archive";
        case WorkspaceId::Warning: return "Warning";
        case WorkspaceId::Volume: return "Volume";
        case WorkspaceId::Tools: return "Tools";
        case WorkspaceId::Assets: return "Assets";
        default: return "Unknown";
    }
}

const char* dockTabLabel(ContextDockTab tab) {
    switch (tab) {
        case ContextDockTab::Inspect: return "Inspect";
        case ContextDockTab::Alerts: return "Alerts";
        case ContextDockTab::Layers: return "Layers";
        case ContextDockTab::Assets: return "Assets";
        case ContextDockTab::Session: return "Session";
        default: return "Unknown";
    }
}

ImVec4 rgbaToImVec4(uint32_t color) {
    return ImVec4(
        (float)(color & 0xFF) / 255.0f,
        (float)((color >> 8) & 0xFF) / 255.0f,
        (float)((color >> 16) & 0xFF) / 255.0f,
        (float)((color >> 24) & 0xFF) / 255.0f);
}

std::string formatBytes(size_t bytes) {
    static const char* kUnits[] = {"B", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        unit++;
    }
    char buffer[32];
    if (unit == 0)
        std::snprintf(buffer, sizeof(buffer), "%llu %s", (unsigned long long)bytes, kUnits[unit]);
    else
        std::snprintf(buffer, sizeof(buffer), "%.2f %s", value, kUnits[unit]);
    return buffer;
}

const char* performanceProfileLabel(PerformanceProfile profile) {
    switch (profile) {
        case PerformanceProfile::Auto: return "Auto";
        case PerformanceProfile::Quality: return "Quality";
        case PerformanceProfile::Balanced: return "Balanced";
        case PerformanceProfile::Performance: return "Performance";
        default: return "Unknown";
    }
}

void resetConusView(App& app) {
    app.viewport().center_lat = 39.0;
    app.viewport().center_lon = -98.0;
    app.viewport().zoom = 28.0;
}

void centerOnWarning(App& app, const WarningPolygon& warning) {
    if (warning.lats.empty() || warning.lons.empty())
        return;
    double latSum = 0.0;
    double lonSum = 0.0;
    const size_t count = std::min(warning.lats.size(), warning.lons.size());
    for (size_t i = 0; i < count; ++i) {
        latSum += warning.lats[i];
        lonSum += warning.lons[i];
    }
    app.viewport().center_lat = latSum / (double)count;
    app.viewport().center_lon = lonSum / (double)count;
    app.viewport().zoom = std::max(app.viewport().zoom, 85.0);
}

void drawLiveLoopTimeline(App& app, const char* id, float height = 26.0f) {
    const int targetFrames = std::max(1, app.liveLoopLength());
    const int availableFrames = std::max(0, app.liveLoopAvailableFrames());
    const int loadedStartSlot = std::max(0, targetFrames - availableFrames);
    const int playbackFrame = std::clamp(app.liveLoopPlaybackFrame(), 0, std::max(0, availableFrames - 1));
    const int currentSlot = (availableFrames > 0)
        ? std::clamp(loadedStartSlot + playbackFrame, 0, targetFrames - 1)
        : targetFrames - 1;
    const int renderPending = std::max(0, app.liveLoopBackfillPendingFrames());
    const int fetchTotal = std::max(0, app.liveLoopBackfillFetchTotal());
    const int fetchDone = std::max(0, app.liveLoopBackfillFetchCompleted());
    const int downloadOutstanding = std::max(0, fetchTotal - fetchDone);

    const float width = std::max(240.0f, ImGui::GetContentRegionAvail().x);
    ImGui::InvisibleButton(id, ImVec2(width, height));
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    ImDrawList* draw = ImGui::GetWindowDrawList();

    const ImU32 bg = IM_COL32(16, 19, 24, 235);
    const ImU32 border = IM_COL32(72, 80, 92, 255);
    const ImU32 missing = IM_COL32(42, 48, 56, 255);
    const ImU32 downloading = IM_COL32(69, 92, 130, 255);
    const ImU32 rendering = IM_COL32(124, 96, 40, 255);
    const ImU32 ready = IM_COL32(48, 168, 104, 255);
    const ImU32 current = IM_COL32(255, 214, 96, 255);
    const ImU32 liveEdge = IM_COL32(78, 220, 136, 255);

    draw->AddRectFilled(min, max, bg, 6.0f);
    draw->AddRect(min, max, border, 6.0f, 0, 1.0f);

    const float innerPad = 4.0f;
    const float trackMinX = min.x + innerPad;
    const float trackMaxX = max.x - innerPad;
    const float trackMinY = min.y + innerPad;
    const float trackMaxY = max.y - innerPad;
    const float slotWidth = (trackMaxX - trackMinX) / (float)targetFrames;

    const int renderStartSlot = std::max(0, loadedStartSlot - std::min(renderPending, loadedStartSlot));
    const int downloadStartSlot = std::max(0, renderStartSlot - std::min(downloadOutstanding, renderStartSlot));

    for (int slot = 0; slot < targetFrames; ++slot) {
        ImU32 color = missing;
        if (slot >= loadedStartSlot) color = ready;
        else if (slot >= renderStartSlot) color = rendering;
        else if (slot >= downloadStartSlot) color = downloading;

        const float x0 = trackMinX + slotWidth * slot;
        const float x1 = (slot == targetFrames - 1) ? trackMaxX : (trackMinX + slotWidth * (slot + 1));
        const float pad = slotWidth > 3.0f ? 0.75f : 0.0f;
        draw->AddRectFilled(ImVec2(x0 + pad, trackMinY), ImVec2(x1 - pad, trackMaxY), color, 2.5f);
    }

    const float currentX = trackMinX + slotWidth * (float)currentSlot;
    draw->AddRectFilled(ImVec2(currentX, min.y + 1.5f),
                        ImVec2(std::min(trackMaxX, currentX + std::max(2.0f, slotWidth)), max.y - 1.5f),
                        current, 3.0f);
    draw->AddLine(ImVec2(trackMaxX - 1.5f, min.y + 3.0f), ImVec2(trackMaxX - 1.5f, max.y - 3.0f),
                  liveEdge, 2.0f);

    auto slotFromMouse = [&](float mouseX) {
        const float normalized = std::clamp((mouseX - trackMinX) / std::max(1.0f, trackMaxX - trackMinX), 0.0f, 0.999999f);
        return std::clamp((int)std::floor(normalized * targetFrames), 0, targetFrames - 1);
    };

    if (ImGui::IsItemHovered()) {
        g_uiWantsMouseCapture = true;
        const int hoveredSlot = slotFromMouse(ImGui::GetIO().MousePos.x);
        ImGui::BeginTooltip();
        ImGui::Text("Frame Slot %d / %d", hoveredSlot + 1, targetFrames);
        if (hoveredSlot >= loadedStartSlot && availableFrames > 0) {
            const int frameIndex = hoveredSlot - loadedStartSlot;
            const std::string label = app.liveLoopLabelAtFrame(frameIndex);
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Ready");
            if (!label.empty())
                ImGui::TextWrapped("%s", label.c_str());
        } else if (hoveredSlot >= renderStartSlot) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.35f, 1.0f), "Queued for render");
        } else if (hoveredSlot >= downloadStartSlot) {
            ImGui::TextColored(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "Downloading source scan");
        } else {
            ImGui::TextDisabled("Not loaded yet");
        }
        ImGui::EndTooltip();
    }

    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) ||
        (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))) {
        g_uiWantsMouseCapture = true;
        if (availableFrames > 0) {
            const int slot = slotFromMouse(ImGui::GetIO().MousePos.x);
            const int frameIndex = (slot < loadedStartSlot)
                ? 0
                : std::clamp(slot - loadedStartSlot, 0, availableFrames - 1);
            app.setLiveLoopPlaybackFrame(frameIndex);
        }
    }
}

Viewport paneViewport(const Viewport& root, const RadarPanelRect& rect) {
    Viewport pane = root;
    pane.width = rect.width;
    pane.height = rect.height;
    return pane;
}

void drawStationMarkers(App& app, const Viewport& vp, const ImVec2& origin,
                        const std::vector<StationUiState>& stations, int activeIdx) {
    if (app.m_historicMode) return;
    auto* drawList = ImGui::GetBackgroundDrawList();
    for (int i = 0; i < (int)stations.size(); i++) {
        const auto& st = stations[i];
        if (!app.showExperimentalSites() && NEXRAD_STATIONS[i].experimental) continue;

        const float px = origin.x + (float)((st.display_lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
        const float py = origin.y + (float)((vp.center_lat - st.display_lat) * vp.zoom + vp.height * 0.5);
        if (px < origin.x - 50 || px > origin.x + vp.width + 50 ||
            py < origin.y - 50 || py > origin.y + vp.height + 50) {
            continue;
        }

        const bool isActive = (i == activeIdx);
        const float boxW = 36.0f;
        const float boxH = 14.0f;
        ImU32 bgCol;
        ImU32 borderCol;
        ImU32 textCol;
        if (!st.enabled) {
            bgCol = isActive ? IM_COL32(95, 95, 105, 220) : IM_COL32(52, 52, 58, 170);
            borderCol = isActive ? IM_COL32(185, 185, 195, 255) : IM_COL32(96, 96, 104, 190);
            textCol = IM_COL32(185, 185, 195, 230);
        } else {
            bgCol = isActive ? IM_COL32(0, 180, 80, 220) : IM_COL32(40, 40, 50, 180);
            borderCol = isActive ? IM_COL32(100, 255, 150, 255) : IM_COL32(80, 80, 100, 200);
            textCol = isActive ? IM_COL32(255, 255, 255, 255) : IM_COL32(180, 180, 200, 220);
        }
        ImVec2 tl(px - boxW * 0.5f, py - boxH * 0.5f);
        ImVec2 br(px + boxW * 0.5f, py + boxH * 0.5f);
        drawList->AddRectFilled(tl, br, bgCol, 3.0f);
        drawList->AddRect(tl, br, borderCol, 3.0f);
        const char* label = st.icao.c_str();
        ImVec2 textSize = ImGui::CalcTextSize(label);
        drawList->AddText(ImVec2(px - textSize.x * 0.5f, py - textSize.y * 0.5f), textCol, label);
    }
}

void drawWarningPolygons(App& app, const Viewport& vp, const ImVec2& origin,
                         const std::vector<WarningPolygon>& warnings) {
    if (warnings.empty() || !app.m_warningOptions.enabled) return;
    auto* drawList = ImGui::GetBackgroundDrawList();
    for (const auto& warning : warnings) {
        if (warning.lats.size() < 3) continue;
        std::vector<ImVec2> pts;
        pts.reserve(warning.lats.size());
        bool anyOnScreen = false;
        for (int i = 0; i < (int)warning.lats.size(); i++) {
            const float sx = origin.x + (float)((warning.lons[i] - vp.center_lon) * vp.zoom + vp.width * 0.5);
            const float sy = origin.y + (float)((vp.center_lat - warning.lats[i]) * vp.zoom + vp.height * 0.5);
            pts.push_back(ImVec2(sx, sy));
            if (sx > origin.x - 100 && sx < origin.x + vp.width + 100 &&
                sy > origin.y - 100 && sy < origin.y + vp.height + 100) {
                anyOnScreen = true;
            }
        }
        if (!anyOnScreen) continue;
        if (app.m_warningOptions.fillPolygons && pts.size() >= 3) {
            drawList->AddConcavePolyFilled(pts.data(), (int)pts.size(),
                                           app.m_warningOptions.resolvedFillColor(warning));
        }
        if (app.m_warningOptions.outlinePolygons) {
            const uint32_t outlineCol = (warning.color & 0x00FFFFFFu) | 0xFF000000u;
            for (int i = 0; i < (int)pts.size(); i++) {
                const int j = (i + 1) % (int)pts.size();
                drawList->AddLine(pts[i], pts[j], outlineCol, warning.line_width);
            }
        }
    }
}

void drawRadarPane(App& app, const Viewport& rootVp, const ImGuiViewport* mainViewport,
                   const std::vector<StationUiState>& stations,
                   const std::vector<WarningPolygon>& warnings, int paneIndex) {
    const RadarPanelRect rect = app.radarPanelRect(paneIndex);
    const Viewport paneVp = paneViewport(rootVp, rect);
    const ImVec2 origin(mainViewport->Pos.x + (float)rect.x, mainViewport->Pos.y + (float)rect.y);
    auto* drawList = ImGui::GetBackgroundDrawList();
    drawList->PushClipRect(origin, ImVec2(origin.x + rect.width, origin.y + rect.height), true);
    app.basemap().drawBase(drawList, paneVp, origin);
    drawList->AddImage((ImTextureID)(uintptr_t)app.panelTexture(paneIndex).textureId(),
                       origin,
                       ImVec2(origin.x + rect.width, origin.y + rect.height));
    app.basemap().drawOverlay(drawList, paneVp, origin);
    drawStationMarkers(app, paneVp, origin, stations, app.activeStation());
    drawWarningPolygons(app, paneVp, origin, warnings);

    char badge[96];
    std::snprintf(badge, sizeof(badge), "%s | T%d",
                  PRODUCT_INFO[app.radarPanelProduct(paneIndex)].name,
                  app.activeTilt() + 1);
    const ImVec2 textSize = ImGui::CalcTextSize(badge);
    const ImVec2 badgeTl(origin.x + 10.0f, origin.y + 10.0f);
    const ImVec2 badgeBr(badgeTl.x + textSize.x + 16.0f, badgeTl.y + textSize.y + 8.0f);
    drawList->AddRectFilled(badgeTl, badgeBr, IM_COL32(8, 12, 18, 190), 4.0f);
    drawList->AddRect(badgeTl, badgeBr,
                      paneIndex == 0 ? IM_COL32(100, 210, 255, 200) : IM_COL32(90, 90, 110, 180),
                      4.0f);
    drawList->AddText(ImVec2(badgeTl.x + 8.0f, badgeTl.y + 4.0f),
                      IM_COL32(230, 236, 244, 240), badge);
    drawList->AddRect(origin, ImVec2(origin.x + rect.width, origin.y + rect.height),
                      IM_COL32(34, 42, 55, 200));
    drawList->PopClipRect();
}

void drawSingleRadarCanvas(App& app, const Viewport& vp, const ImGuiViewport* mainViewport,
                           const std::vector<StationUiState>& stations,
                           const std::vector<WarningPolygon>& warnings) {
    auto* drawList = ImGui::GetBackgroundDrawList();
    app.basemap().drawBase(drawList, vp, mainViewport->Pos);
    drawList->AddImage((ImTextureID)(uintptr_t)app.outputTexture().textureId(),
                       mainViewport->Pos,
                       ImVec2(mainViewport->Pos.x + mainViewport->Size.x,
                              mainViewport->Pos.y + mainViewport->Size.y));
    app.basemap().drawOverlay(drawList, vp, mainViewport->Pos);
    drawStationMarkers(app, vp, mainViewport->Pos, stations, app.activeStation());
    drawWarningPolygons(app, vp, mainViewport->Pos, warnings);
}

void syncWorkspaceFromApp(App& app) {
    if (app.m_historicMode || app.snapshotMode())
        g_shellState.workspace = WorkspaceId::Archive;
    else if (app.mode3D() || app.crossSection())
        g_shellState.workspace = WorkspaceId::Volume;
    else if (app.radarPanelCount() > 1)
        g_shellState.workspace = WorkspaceId::Compare;
    else
        g_shellState.workspace = WorkspaceId::Live;

    if (app.m_historicMode)
        g_shellState.timeMode = TimeDeckMode::Archive;
    else if (app.snapshotMode())
        g_shellState.timeMode = TimeDeckMode::Snapshot;
    else if (app.liveLoopEnabled())
        g_shellState.timeMode = TimeDeckMode::Review;
    else
        g_shellState.timeMode = TimeDeckMode::LiveTail;
}

void applyWorkspace(App& app, WorkspaceId workspace) {
    g_shellState.workspace = workspace;
    if (workspace == WorkspaceId::Compare && app.radarPanelLayout() == RadarPanelLayout::Single)
        app.setRadarPanelLayout(RadarPanelLayout::Dual);
    if (workspace == WorkspaceId::Live && app.radarPanelCount() > 1)
        app.setRadarPanelLayout(RadarPanelLayout::Single);
}
