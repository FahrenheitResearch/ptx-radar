#include "ui.h"
#include "workstation.h"
#include "app.h"
#include "nexrad/products.h"
#include "nexrad/stations.h"
#include "net/warnings.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace ui {
namespace {

bool g_uiWantsMouseCapture = false;

WarningPolygon const* findSelectedWarning(const ConsoleSession& session,
                                          const std::vector<WarningPolygon>& warnings) {
    if (session.alertFocus.selectedAlertId.empty())
        return nullptr;
    for (const auto& warning : warnings) {
        if (warning.id == session.alertFocus.selectedAlertId)
            return &warning;
    }
    return nullptr;
}

const char* warningGroupLabel(WarningGroup group) {
    switch (group) {
        case WarningGroup::Tornado: return "Tornado";
        case WarningGroup::Severe: return "Severe";
        case WarningGroup::Fire: return "Fire";
        case WarningGroup::Flood: return "Flood";
        case WarningGroup::Marine: return "Marine";
        case WarningGroup::Watch: return "Watch";
        case WarningGroup::Statement: return "Statement";
        case WarningGroup::Advisory: return "Advisory";
        case WarningGroup::Other: return "Other";
    }
    return "Other";
}

bool pointInWarning(float lat, float lon, const WarningPolygon& warning) {
    const size_t count = std::min(warning.lats.size(), warning.lons.size());
    if (count < 3)
        return false;

    bool inside = false;
    for (size_t i = 0, j = count - 1; i < count; j = i++) {
        const float yi = warning.lats[i];
        const float xi = warning.lons[i];
        const float yj = warning.lats[j];
        const float xj = warning.lons[j];
        const bool intersects = ((yi > lat) != (yj > lat)) &&
            (lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-6f) + xi);
        if (intersects)
            inside = !inside;
    }
    return inside;
}

void centerOnWarning(App& app, const WarningPolygon& warning) {
    const size_t count = std::min(warning.lats.size(), warning.lons.size());
    if (count < 3)
        return;
    double latSum = 0.0;
    double lonSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        latSum += warning.lats[i];
        lonSum += warning.lons[i];
    }
    app.setViewCenterZoom(latSum / (double)count, lonSum / (double)count,
                          std::max(app.viewport().zoom, 85.0));
}

std::string stationLabelFromId(const std::vector<StationUiState>& stations, int stationId) {
    for (const auto& station : stations) {
        if (station.index == stationId)
            return station.icao;
    }
    return "---";
}

struct ShellRegions {
    ImRect topBar;
    ImRect leftRail;
    ImRect centerCanvas;
    ImRect rightDock;
    ImRect timeDeck;
};

ShellRegions computeRegions(const ImGuiViewport* viewport, bool dockOpen, float leftWidth) {
    constexpr float pad = 12.0f;
    constexpr float topH = 70.0f;
    constexpr float rightW = 356.0f;
    constexpr float bottomH = 116.0f;
    constexpr float gap = 10.0f;

    const ImVec2 pos = viewport->Pos;
    const ImVec2 size = viewport->Size;
    ShellRegions r{};

    r.topBar = ImRect(ImVec2(pos.x + pad, pos.y + pad),
                      ImVec2(pos.x + size.x - pad, pos.y + pad + topH));

    const float bodyTop = r.topBar.Max.y + gap;
    const float bodyBottom = pos.y + size.y - pad - bottomH - gap;
    const float rightWidth = dockOpen ? rightW : 0.0f;
    const float railWidth = std::clamp(leftWidth, 100.0f, 220.0f);
    const float centerLeft = pos.x + pad + railWidth + gap;
    const float centerRight = pos.x + size.x - pad - rightWidth - (dockOpen ? gap : 0.0f);

    r.leftRail = ImRect(ImVec2(pos.x + pad, bodyTop),
                        ImVec2(pos.x + pad + railWidth, bodyBottom));
    r.centerCanvas = ImRect(ImVec2(centerLeft, bodyTop),
                            ImVec2(centerRight, bodyBottom));
    r.rightDock = ImRect(ImVec2(pos.x + size.x - pad - rightWidth, bodyTop),
                         ImVec2(pos.x + size.x - pad, bodyBottom));
    r.timeDeck = ImRect(ImVec2(centerLeft, pos.y + size.y - pad - bottomH),
                        ImVec2(pos.x + size.x - pad, pos.y + size.y - pad));
    return r;
}

void handleRailResize(ConsoleSession& session, const ShellRegions& regions) {
    const float splitterW = 8.0f;
    const ImVec2 pos(regions.leftRail.Max.x - splitterW * 0.5f, regions.leftRail.Min.y);
    const ImVec2 size(splitterW, regions.leftRail.GetHeight());
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoBackground;
    ImGui::Begin("##c3_left_splitter", nullptr, flags);
    ImGui::InvisibleButton("##rail_splitter_hit", size);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        g_uiWantsMouseCapture = true;
    }
    if (ImGui::IsItemActive())
        session.workspaceRailWidth = std::clamp(session.workspaceRailWidth + ImGui::GetIO().MouseDelta.x, 100.0f, 220.0f);
    ImGui::End();
}

void beginFixedWindow(const char* name, const ImRect& rect, ImGuiWindowFlags extra = 0, float alpha = 0.96f) {
    ImGui::SetNextWindowPos(rect.Min, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(rect.GetWidth(), rect.GetHeight()), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(alpha);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoTitleBar |
                             extra;
    ImGui::Begin(name, nullptr, flags);
}

void endFixedWindow() {
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem))
        g_uiWantsMouseCapture = true;
    ImGui::End();
}

void resetConusView(App& app) {
    app.setViewCenterZoom(39.0, -98.0, 28.0);
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

std::string formatBytes(size_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    char buffer[32];
    if (unit == 0)
        std::snprintf(buffer, sizeof(buffer), "%llu %s", (unsigned long long)bytes, units[unit]);
    else
        std::snprintf(buffer, sizeof(buffer), "%.2f %s", value, units[unit]);
    return buffer;
}

void drawTimeline(App& app, ConsoleSession& session, const char* id, float height = 28.0f) {
    const int targetFrames = std::max(1, session.transport.requestedFrames);
    const int available = std::max(0, session.transport.readyFrames);
    const int loadedStart = std::max(0, targetFrames - available);
    const int playback = std::clamp(session.transport.cursorFrame, 0, std::max(0, available - 1));
    const int currentSlot = available > 0 ? std::clamp(loadedStart + playback, 0, targetFrames - 1) : targetFrames - 1;

    const float width = std::max(260.0f, ImGui::GetContentRegionAvail().x);
    ImGui::InvisibleButton(id, ImVec2(width, height));
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    ImDrawList* draw = ImGui::GetWindowDrawList();

    draw->AddRectFilled(min, max, IM_COL32(18, 22, 30, 230), 6.0f);
    draw->AddRect(min, max, IM_COL32(70, 78, 92, 255), 6.0f);

    const float slotWidth = (max.x - min.x - 8.0f) / (float)targetFrames;
    for (int slot = 0; slot < targetFrames; ++slot) {
        ImU32 col = slot >= loadedStart ? IM_COL32(52, 170, 108, 255) : IM_COL32(46, 52, 62, 255);
        const float x0 = min.x + 4.0f + slotWidth * slot;
        const float x1 = (slot == targetFrames - 1) ? (max.x - 4.0f) : (min.x + 4.0f + slotWidth * (slot + 1));
        draw->AddRectFilled(ImVec2(x0 + 1.0f, min.y + 4.0f), ImVec2(x1 - 1.0f, max.y - 4.0f), col, 2.0f);
    }

    const float currentX = min.x + 4.0f + slotWidth * (float)currentSlot;
    draw->AddRectFilled(ImVec2(currentX, min.y + 1.5f), ImVec2(std::min(max.x - 4.0f, currentX + std::max(2.0f, slotWidth)), max.y - 1.5f),
                        IM_COL32(255, 214, 96, 255), 3.0f);

    if (ImGui::IsItemHovered() && available > 0) {
        const float normalized = std::clamp((ImGui::GetIO().MousePos.x - (min.x + 4.0f)) / std::max(1.0f, max.x - min.x - 8.0f), 0.0f, 0.999999f);
        const int slot = std::clamp((int)std::floor(normalized * targetFrames), 0, targetFrames - 1);
        const int frameIndex = slot < loadedStart ? 0 : std::clamp(slot - loadedStart, 0, available - 1);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            transportSeekFrame(app, session, frameIndex);
        }
        g_uiWantsMouseCapture = true;
    }
}

void drawPaneHeader(App& app, const ConsoleSession& session,
                    const std::vector<StationUiState>& stations,
                    ImDrawList* draw, const ImVec2& min, int paneIndex,
                    int product, int tilt, bool activePane) {
    const PaneState& pane = session.panes[paneIndex];
    char header[160];
    std::snprintf(header, sizeof(header), "%s | %s | %s | T%d",
                  paneRoleLabel(pane.role),
                  stationLabelFromId(stations, pane.selection.stationId).c_str(),
                  PRODUCT_INFO[product].name,
                  pane.selection.tilt.sweepIndex + 1);

    const ImVec2 textSize = ImGui::CalcTextSize(header);
    const ImVec2 badgeTl(min.x + 10.0f, min.y + 10.0f);
    const ImVec2 badgeBr(badgeTl.x + textSize.x + 16.0f, badgeTl.y + 34.0f);
    draw->AddRectFilled(badgeTl, badgeBr, IM_COL32(8, 12, 18, 210), 4.0f);
    draw->AddRect(badgeTl, badgeBr,
                  activePane ? IM_COL32(100, 210, 255, 220) : IM_COL32(60, 78, 110, 180),
                  4.0f);
    draw->AddText(ImVec2(badgeTl.x + 8.0f, badgeTl.y + 9.0f), IM_COL32(230, 236, 244, 240), header);

    char links[64];
    std::snprintf(links, sizeof(links), "G%d T%d S%d L%d",
                  pane.links.geo, pane.links.time, pane.links.station, pane.links.tilt);
    draw->AddText(ImVec2(badgeTl.x + 8.0f, badgeTl.y + 38.0f), IM_COL32(132, 145, 165, 220), links);
}

void drawStationMarkers(App& app, const std::vector<StationUiState>& stations,
                        ImDrawList* draw, const Viewport& vp, const ImVec2& origin) {
    if (app.m_historicMode)
        return;

    const int activeIdx = app.activeStation();
    for (int i = 0; i < (int)stations.size(); ++i) {
        const auto& st = stations[i];
        if (!app.showExperimentalSites() && NEXRAD_STATIONS[i].experimental)
            continue;

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

        const ImVec2 tl(px - boxW * 0.5f, py - boxH * 0.5f);
        const ImVec2 br(px + boxW * 0.5f, py + boxH * 0.5f);
        draw->AddRectFilled(tl, br, bgCol, 3.0f);
        draw->AddRect(tl, br, borderCol, 3.0f);
        const char* label = st.icao.c_str();
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        draw->AddText(ImVec2(px - textSize.x * 0.5f, py - textSize.y * 0.5f), textCol, label);
    }
}

void drawWarningPolygons(App& app, const std::vector<WarningPolygon>& warnings,
                         ImDrawList* draw, const Viewport& vp, const ImVec2& origin) {
    if (!app.m_warningOptions.enabled || warnings.empty())
        return;

    for (const auto& warning : warnings) {
        const size_t count = std::min(warning.lats.size(), warning.lons.size());
        if (count < 3)
            continue;

        std::vector<ImVec2> pts;
        pts.reserve(count);
        bool anyOnScreen = false;
        for (size_t i = 0; i < count; ++i) {
            const float sx = origin.x + (float)((warning.lons[i] - vp.center_lon) * vp.zoom + vp.width * 0.5);
            const float sy = origin.y + (float)((vp.center_lat - warning.lats[i]) * vp.zoom + vp.height * 0.5);
            pts.push_back(ImVec2(sx, sy));
            if (sx > origin.x - 100 && sx < origin.x + vp.width + 100 &&
                sy > origin.y - 100 && sy < origin.y + vp.height + 100) {
                anyOnScreen = true;
            }
        }
        if (!anyOnScreen)
            continue;

        if (app.m_warningOptions.fillPolygons)
            draw->AddConcavePolyFilled(pts.data(), (int)pts.size(),
                                       app.m_warningOptions.resolvedFillColor(warning));
        if (app.m_warningOptions.outlinePolygons) {
            const uint32_t outlineCol = (warning.color & 0x00FFFFFFu) | 0xFF000000u;
            for (int i = 0; i < (int)pts.size(); ++i) {
                const int j = (i + 1) % (int)pts.size();
                draw->AddLine(pts[i], pts[j], outlineCol, warning.line_width);
            }
        }
    }
}

void drawCrossSectionLine(App& app, ImDrawList* draw, const Viewport& vp, const ImVec2& origin) {
    if (!app.crossSection())
        return;

    const ImVec2 a(
        origin.x + (float)((app.xsStartLon() - vp.center_lon) * vp.zoom + vp.width * 0.5),
        origin.y + (float)((vp.center_lat - app.xsStartLat()) * vp.zoom + vp.height * 0.5));
    const ImVec2 b(
        origin.x + (float)((app.xsEndLon() - vp.center_lon) * vp.zoom + vp.width * 0.5),
        origin.y + (float)((vp.center_lat - app.xsEndLat()) * vp.zoom + vp.height * 0.5));

    draw->AddLine(a, b, IM_COL32(255, 190, 74, 230), 2.5f);
    draw->AddCircleFilled(a, 4.0f, IM_COL32(255, 214, 102, 255), 16);
    draw->AddCircleFilled(b, 4.0f, IM_COL32(255, 214, 102, 255), 16);
}

void drawCanvas(App& app, ConsoleSession& session, const ShellRegions& regions,
                const ImGuiViewport* mainViewport,
                const std::vector<StationUiState>& stations,
                const std::vector<WarningPolygon>& warnings) {
    const ImRect rect = regions.centerCanvas;
    app.setRadarCanvasRect((int)(rect.Min.x - mainViewport->Pos.x),
                           (int)(rect.Min.y - mainViewport->Pos.y),
                           (int)rect.GetWidth(),
                           (int)rect.GetHeight());

    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    const Viewport root = app.viewport();

    if (app.radarPanelCount() > 1 && !app.mode3D() && !app.crossSection()) {
        for (int pane = 0; pane < app.radarPanelCount(); ++pane) {
            const RadarPanelRect prect = app.radarPanelRect(pane);
            const ImVec2 min(mainViewport->Pos.x + (float)prect.x, mainViewport->Pos.y + (float)prect.y);
            const ImVec2 max(min.x + prect.width, min.y + prect.height);
            const Viewport paneVp = [&]() {
                Viewport vp = root;
                vp.width = prect.width;
                vp.height = prect.height;
                return vp;
            }();
            draw->PushClipRect(min, max, true);
            app.basemap().drawBase(draw, paneVp, min);
            draw->AddImage((ImTextureID)(uintptr_t)app.panelTexture(pane).textureId(), min, max);
            app.basemap().drawOverlay(draw, paneVp, min);
            drawWarningPolygons(app, warnings, draw, paneVp, min);
            drawStationMarkers(app, stations, draw, paneVp, min);
            draw->AddRect(min, max, IM_COL32(35, 42, 56, 180), 0.0f, 0, 1.0f);
            drawPaneHeader(app, session, stations, draw, min, pane,
                           app.radarPanelProduct(pane), app.radarPanelTilt(pane),
                           session.activePaneIndex == pane);
            draw->PopClipRect();
        }
    } else {
        Viewport paneVp = root;
        paneVp.width = (int)rect.GetWidth();
        paneVp.height = (int)rect.GetHeight();
        draw->PushClipRect(rect.Min, rect.Max, true);
        app.basemap().drawBase(draw, paneVp, rect.Min);
        draw->AddImage((ImTextureID)(uintptr_t)app.outputTexture().textureId(), rect.Min, rect.Max);
        app.basemap().drawOverlay(draw, paneVp, rect.Min);
        drawWarningPolygons(app, warnings, draw, paneVp, rect.Min);
        drawStationMarkers(app, stations, draw, paneVp, rect.Min);
        drawCrossSectionLine(app, draw, paneVp, rect.Min);
        draw->AddRect(rect.Min, rect.Max, IM_COL32(35, 42, 56, 180), 0.0f, 0, 1.0f);
        drawPaneHeader(app, session, stations, draw, rect.Min, 0,
                       app.activeProduct(), app.activeTilt(), true);

        const WarningPolygon* selectedWarning = findSelectedWarning(session, warnings);
        if (selectedWarning) {
            const char* label = selectedWarning->event.c_str();
            const ImVec2 textSize = ImGui::CalcTextSize(label);
            const ImVec2 tl(rect.Min.x + 10.0f, rect.Min.y + 62.0f);
            const ImVec2 br(tl.x + textSize.x + 16.0f, tl.y + 24.0f);
            draw->AddRectFilled(tl, br, IM_COL32(44, 16, 16, 210), 4.0f);
            draw->AddRect(tl, br, IM_COL32(196, 78, 78, 220), 4.0f);
            draw->AddText(ImVec2(tl.x + 8.0f, tl.y + 4.0f), IM_COL32(255, 220, 220, 240), label);
        }

        if (app.crossSection()) {
            const float xsHeight = std::max(220.0f, rect.GetHeight() * 0.34f);
            const ImRect xsRect(
                ImVec2(rect.Min.x + 12.0f, rect.Max.y - xsHeight - 12.0f),
                ImVec2(rect.Max.x - 12.0f, rect.Max.y - 12.0f));
            draw->AddRectFilled(xsRect.Min, xsRect.Max, IM_COL32(10, 12, 18, 242), 8.0f);
            draw->AddRect(xsRect.Min, xsRect.Max, IM_COL32(70, 84, 116, 210), 8.0f);
            if (app.xsTexture().textureId() != 0)
                draw->AddImage((ImTextureID)(uintptr_t)app.xsTexture().textureId(),
                               xsRect.Min, xsRect.Max);
            draw->AddText(ImVec2(xsRect.Min.x + 10.0f, xsRect.Min.y + 8.0f),
                          IM_COL32(222, 230, 245, 230), "Cross Section");
        }
        draw->PopClipRect();
    }

    const std::string& attribution = app.basemap().attribution();
    if (!attribution.empty()) {
        const ImVec2 textSize = ImGui::CalcTextSize(attribution.c_str());
        const ImVec2 tl(rect.Min.x + 12.0f, rect.Max.y - textSize.y - 12.0f);
        draw->AddRectFilled(tl, ImVec2(tl.x + textSize.x + 10.0f, tl.y + textSize.y + 6.0f),
                            IM_COL32(6, 10, 16, 168), 4.0f);
        draw->AddText(ImVec2(tl.x + 5.0f, tl.y + 3.0f), IM_COL32(220, 228, 238, 210), attribution.c_str());
    }
}

void renderTopBar(App& app, ConsoleSession& session, const ShellRegions& regions,
                  const std::vector<StationUiState>& stations,
                  const std::vector<WarningPolygon>& warnings) {
    beginFixedWindow("##c3_top_bar", regions.topBar);

    ImGui::TextColored(ImVec4(0.85f, 0.91f, 0.98f, 1.0f), "CURSDAR3");
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("%s", workspaceLabel(session.activeWorkspace));
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("Site %s", stationLabelFromId(stations, session.stationWorkflow.focusedStationId).c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("Alerts %d", (int)warnings.size());
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("Loaded %d", app.stationsLoaded());

    ImGui::Separator();

    if (ImGui::Button(session.stationWorkflow.followNearest ? "Scout" : "Locked")) {
        session.stationWorkflow.followNearest = !session.stationWorkflow.followNearest;
        if (session.stationWorkflow.followNearest) {
            session.stationWorkflow.lockedStationId = -1;
            app.setAutoTrackStation(true);
        } else if (session.stationWorkflow.focusedStationId >= 0) {
            session.stationWorkflow.lockedStationId = session.stationWorkflow.focusedStationId;
            app.setAutoTrackStation(false);
        }
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::InputTextWithHint("##site_search", "Site", session.stationWorkflow.searchQuery, IM_ARRAYSIZE(session.stationWorkflow.searchQuery),
                                 ImGuiInputTextFlags_CharsUppercase | ImGuiInputTextFlags_EnterReturnsTrue)) {
        for (const auto& station : stations) {
            if (_stricmp(station.icao.c_str(), session.stationWorkflow.searchQuery) == 0) {
                session.stationWorkflow.followNearest = false;
                session.stationWorkflow.lockedStationId = station.index;
                app.setAutoTrackStation(false);
                focusStation(app, session, station.index, true, -1.0);
                break;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(session.transport.playIntent ? "Pause" : "Play")) {
        toggleTransportPlay(app, session);
    }
    ImGui::SameLine();
    if (ImGui::Button("Live"))
        transportJumpLive(app, session);
    ImGui::SameLine();
    if (ImGui::Button("Refresh"))
        app.refreshData();
    ImGui::SameLine();
    if (ImGui::Button(app.mode3D() ? "Exit 3D" : "3D")) {
        app.toggle3D();
        session.activeWorkspace = app.mode3D() ? WorkspaceId::Volume : WorkspaceId::Live;
    }
    ImGui::SameLine();
    if (ImGui::Button(app.crossSection() ? "Hide XS" : "Cross Section")) {
        app.toggleCrossSection();
        session.activeWorkspace = app.crossSection() ? WorkspaceId::Volume : WorkspaceId::Live;
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    for (int product = 0; product < (int)Product::COUNT; ++product) {
        if (product == session.panes[session.activePaneIndex].selection.product)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
        if (ImGui::Button(PRODUCT_INFO[product].code))
            setPaneProduct(app, session, session.activePaneIndex, product);
        if (product == session.panes[session.activePaneIndex].selection.product)
            ImGui::PopStyleColor();
        if (product != (int)Product::COUNT - 1)
            ImGui::SameLine();
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    for (int tilt = 0; tilt < app.maxTilts(); ++tilt) {
        char label[8];
        std::snprintf(label, sizeof(label), "T%d", tilt + 1);
        if (tilt == session.panes[session.activePaneIndex].selection.tilt.sweepIndex)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.30f, 0.22f, 1.0f));
        if (ImGui::Button(label))
            setPaneTilt(app, session, session.activePaneIndex, tilt);
        if (tilt == session.panes[session.activePaneIndex].selection.tilt.sweepIndex)
            ImGui::PopStyleColor();
        if (tilt != app.maxTilts() - 1)
            ImGui::SameLine();
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    int layoutIdx = app.radarPanelLayout() == RadarPanelLayout::Single ? 0 :
                    app.radarPanelLayout() == RadarPanelLayout::Dual ? 1 : 2;
    const char* layoutLabels[] = {"Single", "Dual", "Quad"};
    ImGui::SetNextItemWidth(96.0f);
    if (ImGui::Combo("##layout_combo", &layoutIdx, layoutLabels, IM_ARRAYSIZE(layoutLabels))) {
        if (layoutIdx == 0) {
            setWorkspace(app, session, WorkspaceId::Live);
        } else if (layoutIdx == 1) {
            setWorkspace(app, session, WorkspaceId::Compare);
        } else {
            session.activeWorkspace = (session.activeWorkspace == WorkspaceId::Warning)
                ? WorkspaceId::Warning
                : WorkspaceId::Compare;
            app.setRadarPanelLayout(RadarPanelLayout::Quad);
        }
    }

    ImGui::SameLine();
    bool geoLinked = session.panes[session.activePaneIndex].links.geo == session.panes[0].links.geo;
    if (ImGui::Checkbox("Geo", &geoLinked))
        session.panes[session.activePaneIndex].links.geo = geoLinked ? session.panes[0].links.geo : session.activePaneIndex + 1;
    ImGui::SameLine();
    bool timeLinked = session.panes[session.activePaneIndex].links.time == session.panes[0].links.time;
    if (ImGui::Checkbox("Time", &timeLinked))
        session.panes[session.activePaneIndex].links.time = timeLinked ? session.panes[0].links.time : session.activePaneIndex + 1;
    ImGui::SameLine();
    bool stationLinked = session.panes[session.activePaneIndex].links.station == session.panes[0].links.station;
    if (ImGui::Checkbox("Station", &stationLinked))
        session.panes[session.activePaneIndex].links.station = stationLinked ? session.panes[0].links.station : session.activePaneIndex + 1;
    ImGui::SameLine();
    bool tiltLinked = session.panes[session.activePaneIndex].links.tilt == session.panes[0].links.tilt;
    if (ImGui::Checkbox("Tilt", &tiltLinked))
        session.panes[session.activePaneIndex].links.tilt = tiltLinked ? session.panes[0].links.tilt : session.activePaneIndex + 1;

    endFixedWindow();
}

void renderRail(App& app, ConsoleSession& session, const ShellRegions& regions) {
    beginFixedWindow("##c3_workspace_rail", regions.leftRail);
    auto workspaceButton = [&](WorkspaceId id) {
        const bool active = (id == session.activeWorkspace);
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
        if (ImGui::Button(workspaceLabel(id), ImVec2(-1.0f, 34.0f)))
            setWorkspace(app, session, id);
        if (active) ImGui::PopStyleColor();
    };

    workspaceButton(WorkspaceId::Live);
    workspaceButton(WorkspaceId::Compare);
    workspaceButton(WorkspaceId::Warning);

    const bool archiveActive = (session.activeWorkspace == WorkspaceId::Archive) ||
                               (session.contextDockOpen && session.activeDockTab == ContextDockTab::Archive);
    if (archiveActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("Archive", ImVec2(-1.0f, 34.0f))) {
        session.activeWorkspace = WorkspaceId::Archive;
        session.contextDockOpen = true;
        session.activeDockTab = ContextDockTab::Archive;
    }
    if (archiveActive) ImGui::PopStyleColor();

    const bool mode3DActive = app.mode3D();
    if (mode3DActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("3D", ImVec2(-1.0f, 34.0f))) {
        if (app.crossSection())
            app.toggleCrossSection();
        app.toggle3D();
        session.activeWorkspace = app.mode3D() ? WorkspaceId::Volume : WorkspaceId::Live;
    }
    if (mode3DActive) ImGui::PopStyleColor();

    const bool xsActive = app.crossSection();
    if (xsActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("Cross Section", ImVec2(-1.0f, 34.0f))) {
        if (app.mode3D())
            app.toggle3D();
        app.toggleCrossSection();
        session.activeWorkspace = app.crossSection() ? WorkspaceId::Volume : WorkspaceId::Live;
    }
    if (xsActive) ImGui::PopStyleColor();

    const bool toolsActive = (session.activeWorkspace == WorkspaceId::Tools) ||
                             (session.contextDockOpen && session.activeDockTab == ContextDockTab::Tools);
    if (toolsActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("Tools", ImVec2(-1.0f, 34.0f))) {
        session.activeWorkspace = WorkspaceId::Tools;
        session.contextDockOpen = true;
        session.activeDockTab = ContextDockTab::Tools;
    }
    if (toolsActive) ImGui::PopStyleColor();

    const bool layersActive = session.contextDockOpen && session.activeDockTab == ContextDockTab::Layers;
    if (layersActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("Layers", ImVec2(-1.0f, 34.0f))) {
        session.contextDockOpen = true;
        session.activeDockTab = ContextDockTab::Layers;
    }
    if (layersActive) ImGui::PopStyleColor();

    const bool assetsActive = session.contextDockOpen && session.activeDockTab == ContextDockTab::Assets;
    if (assetsActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("Assets", ImVec2(-1.0f, 34.0f))) {
        session.contextDockOpen = true;
        session.activeDockTab = ContextDockTab::Assets;
    }
    if (assetsActive) ImGui::PopStyleColor();

    const bool sessionActive = session.contextDockOpen && session.activeDockTab == ContextDockTab::Session;
    if (sessionActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
    if (ImGui::Button("Session", ImVec2(-1.0f, 34.0f))) {
        session.contextDockOpen = true;
        session.activeDockTab = ContextDockTab::Session;
    }
    if (sessionActive) ImGui::PopStyleColor();

    ImGui::Separator();
    if (ImGui::Button("CONUS", ImVec2(-1.0f, 32.0f)))
        resetConusView(app);
    if (ImGui::Button("Show All", ImVec2(-1.0f, 32.0f)))
        app.toggleShowAll();

    endFixedWindow();
}

void renderDock(App& app, ConsoleSession& session, const ShellRegions& regions,
                const std::vector<StationUiState>& stations,
                const std::vector<WarningPolygon>& warnings) {
    if (!session.contextDockOpen)
        return;

    static char archiveStation[16] = "KTLX";
    static int archiveYear = 2025;
    static int archiveMonth = 3;
    static int archiveDay = 30;
    static int archiveStartHour = 21;
    static int archiveStartMin = 0;
    static int archiveEndHour = 22;
    static int archiveEndMin = 0;
    static char pollingUrl[512] = "";
    static char palettePath[512] = "";
    static std::string assetStatus;

    beginFixedWindow("##c3_context_dock", regions.rightDock);
    auto dockTabButton = [&](const char* label, ContextDockTab tab) {
        const bool active = session.activeDockTab == tab;
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.34f, 0.48f, 1.0f));
        if (ImGui::Button(label))
            session.activeDockTab = tab;
        if (active) ImGui::PopStyleColor();
    };

    dockTabButton("Inspect", ContextDockTab::Inspect);
    ImGui::SameLine();
    dockTabButton("Alerts", ContextDockTab::Alerts);
    ImGui::SameLine();
    dockTabButton("Archive", ContextDockTab::Archive);
    ImGui::SameLine();
    dockTabButton("Tools", ContextDockTab::Tools);
    ImGui::SameLine();
    dockTabButton("Layers", ContextDockTab::Layers);
    ImGui::SameLine();
    dockTabButton("Assets", ContextDockTab::Assets);
    ImGui::SameLine();
    dockTabButton("Session", ContextDockTab::Session);
    ImGui::Separator();

    if (session.activeDockTab == ContextDockTab::Inspect) {
            const PaneState& pane = session.panes[session.activePaneIndex];
            ImGui::Text("Focused Station: %s", stationLabelFromId(stations, session.stationWorkflow.focusedStationId).c_str());
            ImGui::Text("Hover Station: %s", stationLabelFromId(stations, session.stationWorkflow.hoveredStationId).c_str());
            ImGui::Text("Locked Station: %s", stationLabelFromId(stations, session.stationWorkflow.lockedStationId).c_str());
            ImGui::Separator();
            ImGui::Text("Pane %d | %s", session.activePaneIndex + 1, paneRoleLabel(pane.role));
            ImGui::Text("Product: %s", PRODUCT_INFO[pane.selection.product].name);
            ImGui::Text("Tilt: %.1f", pane.selection.tilt.elevationDeg);
            ImGui::Text("Loaded: %d / %d", app.stationsLoaded(), app.stationsTotal());
            const auto& mem = app.memoryTelemetry();
            ImGui::Separator();
            ImGui::Text("VRAM %s / %s", formatBytes(mem.gpu_used_bytes).c_str(), formatBytes(mem.gpu_total_bytes).c_str());
            ImGui::Text("RAM %s", formatBytes(mem.process_working_set_bytes).c_str());
            if (session.stationWorkflow.focusedStationId >= 0 && session.stationWorkflow.focusedStationId < (int)stations.size()) {
                const auto& st = stations[session.stationWorkflow.focusedStationId];
                if (!st.latest_scan_utc.empty())
                    ImGui::Text("Latest scan: %s", st.latest_scan_utc.c_str());
            }
    } else if (session.activeDockTab == ContextDockTab::Alerts) {
            ImGui::Checkbox("Overlays", &app.m_warningOptions.enabled);
            ImGui::Checkbox("Warnings", &app.m_warningOptions.showWarnings);
            ImGui::SameLine();
            ImGui::Checkbox("Watches", &app.m_warningOptions.showWatches);
            ImGui::SameLine();
            ImGui::Checkbox("Statements", &app.m_warningOptions.showStatements);
            ImGui::Separator();
            if (warnings.empty()) {
                ImGui::TextDisabled("No alert polygons loaded.");
            } else {
                for (size_t i = 0; i < warnings.size(); ++i) {
                    const bool selected = warnings[i].id == session.alertFocus.selectedAlertId;
                    const std::string alertLabel = warnings[i].event + "##alert_" + std::to_string(i);
                    if (ImGui::Selectable(alertLabel.c_str(), selected)) {
                        selectAlert(app, session, warnings, stations, warnings[i].id);
                        centerOnWarning(app, warnings[i]);
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", warningGroupLabel(warnings[i].group));
                    if (!warnings[i].headline.empty())
                        ImGui::TextWrapped("%s", warnings[i].headline.c_str());
                    if (selected) {
                        if (ImGui::Button("Focus")) {
                            centerOnWarning(app, warnings[i]);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Interrogate Tornado")) {
                            activateTornadoInterrogate(app, session, warnings, stations);
                        }
                        if (!session.alertFocus.candidateStations.empty()) {
                            ImGui::SeparatorText("Candidate Stations");
                            for (int candidateIdx = 0; candidateIdx < (int)session.alertFocus.candidateStations.size(); ++candidateIdx) {
                                const int stationId = session.alertFocus.candidateStations[candidateIdx];
                                const std::string stationButton =
                                    stationLabelFromId(stations, stationId) + "##candidate_" + std::to_string(candidateIdx);
                                if (ImGui::Button(stationButton.c_str(), ImVec2(88.0f, 0.0f))) {
                                    session.stationWorkflow.followNearest = false;
                                    session.stationWorkflow.lockedStationId = stationId;
                                    app.setAutoTrackStation(false);
                                    focusStation(app, session, stationId, true, 120.0);
                                }
                                ImGui::SameLine();
                            }
                            ImGui::NewLine();
                        }
                    }
                    ImGui::Separator();
                }
            }
    } else if (session.activeDockTab == ContextDockTab::Archive) {
            ImGui::Text("Archive and Snapshot");
            ImGui::Separator();
            if (ImGui::Button("Return Live", ImVec2(-1.0f, 0.0f))) {
                app.refreshData();
                session.activeWorkspace = WorkspaceId::Live;
            }
            if (ImGui::Button("March 30 2025 Snapshot", ImVec2(-1.0f, 0.0f))) {
                app.loadMarch302025Snapshot(false);
                session.activeWorkspace = WorkspaceId::Archive;
            }
            if (ImGui::Button("March 30 2025 Lowest Sweep", ImVec2(-1.0f, 0.0f))) {
                app.loadMarch302025Snapshot(true);
                session.activeWorkspace = WorkspaceId::Archive;
            }

            ImGui::SeparatorText("Historic Events");
            for (int i = 0; i < NUM_HISTORIC_EVENTS; ++i) {
                const HistoricEvent& event = HISTORIC_EVENTS[i];
                const std::string label = std::string(event.name) + "##historic_" + std::to_string(i);
                if (ImGui::Selectable(label.c_str(), false)) {
                    app.loadHistoricEvent(i);
                    session.activeWorkspace = WorkspaceId::Archive;
                }
            }

            ImGui::SeparatorText("Custom Range");
            ImGui::InputText("Station", archiveStation, IM_ARRAYSIZE(archiveStation),
                             ImGuiInputTextFlags_CharsUppercase);
            ImGui::InputInt("Year", &archiveYear);
            ImGui::InputInt("Month", &archiveMonth);
            ImGui::InputInt("Day", &archiveDay);
            ImGui::InputInt("Start Hour", &archiveStartHour);
            ImGui::InputInt("Start Min", &archiveStartMin);
            ImGui::InputInt("End Hour", &archiveEndHour);
            ImGui::InputInt("End Min", &archiveEndMin);
            archiveMonth = std::clamp(archiveMonth, 1, 12);
            archiveDay = std::clamp(archiveDay, 1, 31);
            archiveStartHour = std::clamp(archiveStartHour, 0, 23);
            archiveEndHour = std::clamp(archiveEndHour, 0, 23);
            archiveStartMin = std::clamp(archiveStartMin, 0, 59);
            archiveEndMin = std::clamp(archiveEndMin, 0, 59);
            if (ImGui::Button("Load Range", ImVec2(-1.0f, 0.0f))) {
                app.loadArchiveRange(archiveStation, archiveYear, archiveMonth, archiveDay,
                                     archiveStartHour, archiveStartMin, archiveEndHour, archiveEndMin);
                session.activeWorkspace = WorkspaceId::Archive;
            }
            ImGui::TextDisabled("Loaded %d / %d", app.m_historic.downloadedFrames(), app.m_historic.totalFrames());
            const std::string archiveError = app.m_historic.lastError();
            if (!archiveError.empty())
                ImGui::TextWrapped("%s", archiveError.c_str());

            ImGui::SeparatorText("Projection");
            int projectionIdx = (int)app.archiveProjectionKind();
            const char* projectionLabels[] = {"Volume Timeline", "Sweep Stream"};
            if (ImGui::Combo("Archive Source", &projectionIdx, projectionLabels, IM_ARRAYSIZE(projectionLabels)))
                app.setArchiveProjectionKind((ArchiveProjectionKind)projectionIdx);

            if (app.archiveProjectionKind() == ArchiveProjectionKind::SweepStream) {
                const auto& point = app.archiveInterrogationPoint();
                ImGui::TextDisabled("Default: smooth sub-1.5 degree sweeps with the active product.");
                if (ImGui::Button(app.archiveSweepPointPickArmed() ? "Click Map To Set Point" : "Pick Point On Map",
                                  ImVec2(-1.0f, 0.0f))) {
                    app.armArchiveSweepPointPick(!app.archiveSweepPointPickArmed());
                }
                if (ImGui::Button("Use Cursor Position", ImVec2(-1.0f, 0.0f))) {
                    app.setArchiveInterrogationPoint(app.cursorLat(), app.cursorLon());
                }
                if (point.valid) {
                    ImGui::Text("Point %.3f, %.3f", point.lat, point.lon);
                } else {
                    ImGui::TextDisabled("No interrogation point selected (using sweep-only ordering)");
                }

                SweepFilter filter = app.archiveSweepFilter();
                bool filterChanged = false;
                int styleIdx = (int)filter.style;
                const char* styleLabels[] = {"Dense", "Smooth", "Strict Smooth"};
                if (ImGui::Combo("Sweep Style", &styleIdx, styleLabels, IM_ARRAYSIZE(styleLabels))) {
                    filter.style = (SweepStreamStyle)styleIdx;
                    filterChanged = true;
                }
                filterChanged |= ImGui::Checkbox("Sub-1.5 Only", &filter.sub_1p5_only);
                filterChanged |= ImGui::Checkbox("Require Active Product At Point", &filter.require_active_product);
                filterChanged |= ImGui::Checkbox("Require Any Point Coverage", &filter.require_point_coverage);
                bool limitBeamHeight = filter.max_beam_height_arl_m >= 0.0f;
                if (ImGui::Checkbox("Limit Beam Height", &limitBeamHeight)) {
                    filter.max_beam_height_arl_m = limitBeamHeight
                        ? std::max(1500.0f, filter.max_beam_height_arl_m)
                        : -1.0f;
                    filterChanged = true;
                }
                if (limitBeamHeight) {
                    float beamHeightKm = filter.max_beam_height_arl_m / 1000.0f;
                    if (ImGui::SliderFloat("Max Beam Height (km)", &beamHeightKm, 0.1f, 10.0f, "%.1f")) {
                        filter.max_beam_height_arl_m = beamHeightKm * 1000.0f;
                        filterChanged = true;
                    }
                }
                bool limitElevation = filter.max_elevation_deg >= 0.0f;
                if (ImGui::Checkbox("Limit Elevation", &limitElevation)) {
                    filter.max_elevation_deg = limitElevation
                        ? std::max(1.5f, filter.max_elevation_deg)
                        : -1.0f;
                    filterChanged = true;
                }
                if (limitElevation) {
                    float maxElevation = filter.max_elevation_deg;
                    if (ImGui::SliderFloat("Max Elevation (deg)", &maxElevation, 0.1f, 10.0f, "%.1f")) {
                        filter.max_elevation_deg = maxElevation;
                        filterChanged = true;
                    }
                }
                if (filterChanged)
                    app.setArchiveSweepFilter(filter);

                const auto& timeline = app.archiveSweepTimeline();
                ImGui::SeparatorText("Sweep Stream");
                ImGui::Text("Kept %d / %d sweeps", (int)timeline.frames.size(), timeline.candidate_frames);
                ImGui::TextDisabled("%s", timeline.complete ? "Archive load complete" : "Timeline grows as frames load");
                if (!timeline.frames.empty()) {
                    ImGui::TextWrapped("Current: %s", app.transportSnapshot().current_label.c_str());
                }
            }
    } else if (session.activeDockTab == ContextDockTab::Tools) {
            bool srv = app.srvMode();
            if (ImGui::Checkbox("Storm Relative Velocity", &srv))
                app.toggleSRV();
            float stormSpeed = app.stormSpeed();
            if (ImGui::SliderFloat("Storm Speed", &stormSpeed, 0.0f, 40.0f, "%.1f m/s"))
                app.setStormMotion(stormSpeed, app.stormDir());
            float stormDir = app.stormDir();
            if (ImGui::SliderFloat("Storm Dir", &stormDir, 0.0f, 360.0f, "%.0f deg"))
                app.setStormMotion(app.stormSpeed(), stormDir);
            ImGui::Checkbox("Dealias Velocity", &app.m_dealias);
            float threshold = app.dbzMinThreshold();
            if (ImGui::SliderFloat(app.activeProduct() == PROD_VEL ? "Velocity Threshold" : "Reflectivity Threshold",
                                   &threshold, 0.0f, app.activeProduct() == PROD_VEL ? 80.0f : 80.0f, "%.1f"))
                app.setDbzMinThreshold(threshold);
            if (ImGui::Button("Refresh Data", ImVec2(-1.0f, 0.0f)))
                app.refreshData();
            ImGui::TextWrapped("%s", app.priorityStatus().c_str());
    } else if (session.activeDockTab == ContextDockTab::Layers) {
            int basemapIdx = (int)app.basemap().style();
            const char* basemapLabels[] = {"Relief", "Ops Dark", "Satellite", "Satellite Hybrid"};
            if (ImGui::Combo("Basemap", &basemapIdx, basemapLabels, IM_ARRAYSIZE(basemapLabels)))
                app.basemap().setStyle((BasemapStyle)basemapIdx);
            float rasterOpacity = app.basemap().rasterOpacity();
            if (ImGui::SliderFloat("Raster", &rasterOpacity, 0.0f, 1.0f, "%.2f"))
                app.basemap().setRasterOpacity(rasterOpacity);
            float overlayOpacity = app.basemap().overlayOpacity();
            if (ImGui::SliderFloat("Overlay", &overlayOpacity, 0.0f, 1.0f, "%.2f"))
                app.basemap().setOverlayOpacity(overlayOpacity);
            bool grid = app.basemap().showGrid();
            if (ImGui::Checkbox("Grid", &grid))
                app.basemap().setShowGrid(grid);
            bool statesOn = app.basemap().showStateLines();
            if (ImGui::Checkbox("State Lines", &statesOn))
                app.basemap().setShowStateLines(statesOn);
            bool citiesOn = app.basemap().showCityLabels();
            if (ImGui::Checkbox("Cities", &citiesOn))
                app.basemap().setShowCityLabels(citiesOn);
    } else if (session.activeDockTab == ContextDockTab::Assets) {
            int profileIdx = (int)app.requestedPerformanceProfile();
            const char* profileLabels[] = {"Auto", "Quality", "Balanced", "Performance"};
            if (ImGui::Combo("Profile", &profileIdx, profileLabels, IM_ARRAYSIZE(profileLabels)))
                app.setPerformanceProfile((PerformanceProfile)profileIdx);
            ImGui::TextDisabled("Effective: %s", performanceProfileLabel(app.effectivePerformanceProfile()));
            bool showExperimental = app.showExperimentalSites();
            if (ImGui::Checkbox("Experimental Sites", &showExperimental))
                app.setShowExperimentalSites(showExperimental);
            ImGui::SeparatorText("Color Tables");
            ImGui::InputText("Palette Path", palettePath, IM_ARRAYSIZE(palettePath));
            if (ImGui::Button("Load Palette", ImVec2(-1.0f, 0.0f))) {
                assetStatus = app.loadColorTableFromFile(palettePath) ? app.colorTableStatus() : app.colorTableStatus();
            }
            if (ImGui::Button("Reset Active Palette", ImVec2(-1.0f, 0.0f))) {
                app.resetColorTable(app.activeProduct());
                assetStatus = app.colorTableStatus();
            }
            ImGui::SeparatorText("Polling Links");
            ImGui::InputText("URL", pollingUrl, IM_ARRAYSIZE(pollingUrl));
            if (ImGui::Button("Add Polling Link", ImVec2(-1.0f, 0.0f))) {
                std::string error;
                if (app.m_pollingLinks.addLink(pollingUrl, error)) {
                    assetStatus = "Added polling link";
                    pollingUrl[0] = '\0';
                } else {
                    assetStatus = error.empty() ? "Failed to add polling link" : error;
                }
            }
            if (ImGui::Button("Refresh Links", ImVec2(-1.0f, 0.0f)))
                app.m_pollingLinks.refreshAll();
            const auto links = app.m_pollingLinks.entries();
            for (size_t i = 0; i < links.size(); ++i) {
                ImGui::Separator();
                ImGui::TextWrapped("%s", links[i].title.empty() ? links[i].url.c_str() : links[i].title.c_str());
                ImGui::TextDisabled("%s", links[i].last_status.c_str());
                if (ImGui::SmallButton((std::string("Remove##poll_") + std::to_string(i)).c_str())) {
                    app.m_pollingLinks.removeLink(i);
                    break;
                }
            }
            if (!assetStatus.empty())
                ImGui::TextWrapped("%s", assetStatus.c_str());
            ImGui::TextWrapped("%s", app.priorityStatus().c_str());
    } else if (session.activeDockTab == ContextDockTab::Session) {
            if (ImGui::Button("Solo Live", ImVec2(-1.0f, 28.0f))) {
                setWorkspace(app, session, WorkspaceId::Live);
            }
            if (ImGui::Button("Dual Linked", ImVec2(-1.0f, 28.0f))) {
                setWorkspace(app, session, WorkspaceId::Compare);
            }
            if (ImGui::Button("Tornado Interrogate", ImVec2(-1.0f, 28.0f))) {
                activateTornadoInterrogate(app, session, warnings, stations);
            }
            ImGui::TextDisabled("Workspaces are now shell state, not just panel toggles.");
    }
    endFixedWindow();
}

void renderTimeDeck(App& app, ConsoleSession& session, const ShellRegions& regions) {
    beginFixedWindow("##c3_time_deck", regions.timeDeck);
    ImGui::Text("%s", timeDeckModeLabel(session.transport.mode));
    ImGui::Separator();

    if (session.transport.stream == StreamKind::Archive) {
        const int total = app.transportSnapshot().total_frames;
        int frame = session.transport.cursorFrame;
        if (ImGui::Button(session.transport.playIntent ? "Pause" : "Play")) {
            toggleTransportPlay(app, session);
        }
        ImGui::SameLine();
        if (ImGui::Button("Prev"))
            transportStep(app, session, -1);
        ImGui::SameLine();
        if (ImGui::Button("Next"))
            transportStep(app, session, 1);
        ImGui::SameLine();
        ImGui::Text("%s", session.transport.currentLabel.c_str());
        if (ImGui::SliderInt("##historic_frame", &frame, 0, std::max(0, total - 1), "Frame %d"))
            transportSeekFrame(app, session, frame);
    } else {
        bool liveLoop = session.transport.mode != TimeDeckMode::LiveTail;
        if (ImGui::Checkbox("Loop", &liveLoop))
            transportSetLoopEnabled(app, session, liveLoop);
        ImGui::SameLine();
        if (ImGui::Button(session.transport.playIntent ? "Pause" : "Play")) {
            toggleTransportPlay(app, session);
        }
        ImGui::SameLine();
        if (ImGui::Button("Live"))
            transportJumpLive(app, session);
        ImGui::SameLine();
        if (ImGui::Button("Clear"))
            app.clearLiveLoop();
        ImGui::SameLine();
        int frames = session.transport.requestedFrames;
        ImGui::SetNextItemWidth(110.0f);
        if (ImGui::SliderInt("Frames", &frames, 1, app.liveLoopMaxFrames()))
            transportSetRequestedFrames(app, session, frames);
        ImGui::SameLine();
        float speed = session.transport.rateFps;
        ImGui::SetNextItemWidth(110.0f);
        if (ImGui::SliderFloat("FPS", &speed, 1.0f, 15.0f, "%.0f"))
            transportSetRate(app, session, speed);

        drawTimeline(app, session, "##c3_timeline");
        ImGui::Text("Ready %d / %d", session.transport.readyFrames, session.transport.requestedFrames);
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("Downloads %d / %d", app.liveLoopBackfillFetchCompleted(), app.liveLoopBackfillFetchTotal());
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("Rendering %d", app.liveLoopBackfillPendingFrames());
        if (session.transport.playIntent && !session.transport.isAdvancing && session.transport.loadingFrames > 0) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.46f, 1.0f), "Pending");
        }
    }

    if (!session.transport.bookmarks.empty()) {
        ImGui::Separator();
        for (const auto& bookmark : session.transport.bookmarks) {
            ImGui::BulletText("%s  %s", bookmark.label.c_str(), bookmark.timeText.c_str());
        }
    }

    endFixedWindow();
}

void applyKeyboardShortcuts(App& app) {
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    for (int i = 0; i < (int)Product::COUNT; ++i) {
        if (ImGui::IsKeyPressed((ImGuiKey)(ImGuiKey_1 + i)))
            app.setProduct(i);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) app.prevProduct();
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) app.nextProduct();
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) app.nextTilt();
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) app.prevTilt();
    if (ImGui::IsKeyPressed(ImGuiKey_V)) app.toggle3D();
    if (ImGui::IsKeyPressed(ImGuiKey_X)) app.toggleCrossSection();
    if (ImGui::IsKeyPressed(ImGuiKey_A)) app.toggleShowAll();
    if (ImGui::IsKeyPressed(ImGuiKey_R)) app.refreshData();
    if (ImGui::IsKeyPressed(ImGuiKey_S)) app.toggleSRV();
    if (ImGui::IsKeyPressed(ImGuiKey_Home)) resetConusView(app);
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) app.setAutoTrackStation(true);
}

void handleCanvasInteractions(App& app, ConsoleSession& session,
                              const std::vector<StationUiState>& stations,
                              const std::vector<WarningPolygon>& warnings,
                              const ImGuiViewport* mainViewport) {
    if (g_uiWantsMouseCapture)
        return;

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    updateHoveredStation(session, app.stationAtScreen(mouse.x, mouse.y));
    session.alertFocus.hoveredAlertId.clear();
    int hoveredPane = 0;
    if (app.radarPanelCount() > 1 && !app.mode3D() && !app.crossSection()) {
        for (int pane = 0; pane < app.radarPanelCount(); ++pane) {
            const RadarPanelRect rect = app.radarPanelRect(pane);
            const ImRect paneRect(
                ImVec2(mainViewport->Pos.x + (float)rect.x, mainViewport->Pos.y + (float)rect.y),
                ImVec2(mainViewport->Pos.x + (float)rect.x + rect.width,
                       mainViewport->Pos.y + (float)rect.y + rect.height));
            if (paneRect.Contains(mouse)) {
                hoveredPane = pane;
                break;
            }
        }
    }

    for (const auto& warning : warnings) {
        if (pointInWarning(app.cursorLat(), app.cursorLon(), warning)) {
            session.alertFocus.hoveredAlertId = warning.id;
            break;
        }
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (app.archiveSweepPointPickArmed()) {
            app.setArchiveInterrogationPoint(app.cursorLat(), app.cursorLon());
            return;
        }
        session.activePaneIndex = hoveredPane;
        applySelectedPaneToApp(app, session);

        const int hitIdx = app.stationAtScreen(mouse.x, mouse.y);
        if (hitIdx >= 0 && session.alertFocus.hoveredAlertId.empty()) {
            session.stationWorkflow.followNearest = false;
            session.stationWorkflow.lockedStationId = hitIdx;
            app.setAutoTrackStation(false);
            focusStation(app, session, hitIdx, false, -1.0);
        }

        for (const auto& warning : warnings) {
            if (pointInWarning(app.cursorLat(), app.cursorLon(), warning)) {
                selectAlert(app, session, warnings, stations, warning.id);
                centerOnWarning(app, warning);
                break;
            }
        }
    }

    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        const WarningPolygon* selected = findSelectedWarning(session, warnings);
        if (selected)
            activateTornadoInterrogate(app, session, warnings, stations);
    }
}

} // namespace

void init() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FramePadding = ImVec2(10, 6);
    style.ItemSpacing = ImVec2(8, 8);
    style.WindowPadding = ImVec2(12, 12);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.050f, 0.055f, 0.065f, 0.96f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.070f, 0.080f, 0.100f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.090f, 0.110f, 0.145f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.095f, 0.110f, 0.135f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.130f, 0.155f, 0.190f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.110f, 0.130f, 0.165f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.170f, 0.210f, 0.280f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.200f, 0.255f, 0.350f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.120f, 0.145f, 0.185f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.180f, 0.225f, 0.300f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.090f, 0.102f, 0.124f, 1.0f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.170f, 0.210f, 0.280f, 1.0f);
}

void render(App& app) {
    g_uiWantsMouseCapture = false;
    static ConsoleSession session = defaultConsoleSession();

    ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    const auto stations = app.stations();
    const auto warnings = app.currentWarnings();
    syncConsoleSessionFromApp(app, stations, warnings, session);
    const ShellRegions regions = computeRegions(mainViewport, session.contextDockOpen, session.workspaceRailWidth);
    handleRailResize(session, regions);

    drawCanvas(app, session, regions, mainViewport, stations, warnings);
    renderTopBar(app, session, regions, stations, warnings);
    renderRail(app, session, regions);
    renderDock(app, session, regions, stations, warnings);
    renderTimeDeck(app, session, regions);
    applyKeyboardShortcuts(app);
    handleCanvasInteractions(app, session, stations, warnings, mainViewport);
}

void shutdown() {
}

bool wantsMouseCapture() {
    return g_uiWantsMouseCapture;
}

} // namespace ui
