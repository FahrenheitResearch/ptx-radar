# Cursdar3 redesign brief

## One-sentence vision
Cursdar3 should feel like a modern severe-weather operations console: the radar is always the hero, the hot path is always one click away, and every advanced feature appears exactly when it matters instead of living in a permanent panel graveyard.

## What Cursdar3 is trying to fix
Cursdar2 has the right capabilities but the wrong information architecture. Too many features currently live as persistent panels or collapsible sections. The result is feature abundance without workflow hierarchy.

Cursdar3 should not be "Cursdar2 but prettier." It should be a task-driven workstation built around five ideas:

1. **Focus first** — most of the screen belongs to radar data, not chrome.
2. **Time is global** — live loop, archive playback, and snapshots use one transport system.
3. **Context over clutter** — controls appear in the right place for the current task.
4. **Compare on demand** — multi-pane is a workspace, not a pile of extra settings.
5. **Analyst-grade depth** — tornado interrogation, warnings, cross-sections, and data sources are fast to reach without being always visible.

---

## Core product philosophy
Think of Cursdar3 as a blend of these two UX philosophies:

- **RadarScope philosophy**: the hot path is extremely simple — pick radar, loop, change product, compare quickly.
- **AWIPS/analyst philosophy**: when needed, the operator can open deeper tools, saved displays, panes, overlays, and diagnostic workflows.

Cursdar3 should merge those into one rule:

> The first 90% of actions should feel like RadarScope.
> The last 10% should feel like an elite workstation.

---

## The Cursdar3 UI constitution
These are the non-negotiable design rules for the coding agent.

### Rule 1: No new permanent window by default
A feature does **not** get its own always-on panel unless it is part of one of the five permanent regions:
- top command bar
- left workspace rail
- center radar canvas
- right context dock
- bottom time deck

### Rule 2: Every feature must belong to one of five homes
Every capability must live in exactly one of these homes:
- **Workspace** — changes the main operating mode
- **Context Dock tab** — shows details/settings for the selected thing
- **Time Deck popover** — anything related to loop/playback/time range
- **Asset Manager** — palettes, placefiles/polling links, archive downloads, feed config
- **Command Palette** — infrequent or searchable actions

### Rule 3: Duplicate controls are forbidden
If a control exists in the top bar, it should not also live in the inspector or some side panel unless the duplicate has a clearly different purpose.

### Rule 4: The hot path is sacred
The following must always be reachable immediately without opening drawers:
- station focus / lock
- product
- tilt
- play/pause
- loop/live tail
- pane layout / compare mode
- alerts count / alert focus

### Rule 5: Advanced controls hide until invited
Performance telemetry, color table management, archive download forms, polling-link management, and deep warning styling should not occupy prime screen real estate during active interrogation.

### Rule 6: Time is a first-class object
Live loop and historic playback must share a single mental model, UI component, and state machine.

### Rule 7: Panes are views, not worlds
Each pane is just a view onto a shared scene/time/station context with explicit link toggles.

### Rule 8: Radar colors own the visual drama
UI chrome must stay quiet and neutral so reflectivity, velocity, CC, and warning colors dominate attention.

---

## New screen architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ CURSDAR3  [Live] [Compare] [Archive] [Warning] [Volume]    Site KTLX   Age 0:32   Alerts 3 │
│ Search / Station Picker | Product Chips | Tilt Ladder | Layout | Link Geo/Time/Tilt | HUD  │
├──── Workspace Rail ────┬──────────────────────── Radar Canvas ───────────────────────┬──────┤
│ Live                   │                                                             │ Dock │
│ Compare                │                                                             │      │
│ Archive                │                     Main radar view                          │ Alerts│
│ Warning                │                                                             │ Layers│
│ Volume/XS              │                                                             │ Probe │
│ Tools                  │                                                             │ Asset │
│ Assets                 │                                                             │ Notes │
├────────────────────────┴─────────────────────────────────────────────────────────────┴──────┤
│ Time Deck: [Live Tail] [Play/Pause] [Step] [FPS] [Loop Length] [Frame Strip / markers]     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Permanent regions

#### 1. Top command bar
This replaces most of the current left-console hot-path controls.

Always visible:
- workspace tabs/modes
- station search + nearest/lock toggle
- product chip row
- tilt ladder / sweep selector
- compare layout switcher
- link toggles (geo / time / tilt / station)
- scan age and ingest badge
- alert badge
- compact performance badge (optional, collapsed by default)

#### 2. Left workspace rail
This is navigation, not settings.

Recommended items:
- **Live**
- **Compare**
- **Archive**
- **Warning**
- **Volume / Cross-Section**
- **Tools**
- **Assets**

It should be icon-first, label-on-hover or expandable.

#### 3. Center radar canvas
This stays dominant. Everything else exists to support it.

Canvas overlays should include:
- product legend
- scan timestamp / age badge
- cursor probe mini readout
- station hover cards
- linked crosshair for compare panes
- warning/detection chips when selected

#### 4. Right context dock
This replaces the current trio of separate Station Browser / Inspector / Warnings windows.

Tabs should be:
- **Inspect** — selected site / pane / cursor / warning details
- **Alerts** — warning stack + quick actions
- **Layers** — basemap, warning overlays, detections, placefiles
- **Assets** — palettes, polling links, archive downloads, data sources
- **Session** — saved layouts, bookmarks, future evidence board

The dock should be width-resizable and collapsible.

#### 5. Bottom Time Deck
This replaces both the floating live loop bar and the separate historic timeline panel.

Always the same component, with mode chips for:
- Live Tail
- Review
- Archive
- Snapshot

Controls:
- play/pause
- step forward/back
- jump live
- speed
- loop length / time span
- frame strip with readiness markers
- backfill/download progress
- bookmarks for warning issuance moments / detections / notes

Crucial behavior:
- **changing product or tilt must not kill playback state**
- if new frames are not yet ready, the deck shows pending state but preserves play intent
- when data arrives, playback resumes automatically if play was active

---

## Workspaces

### 1. Live workspace
For day-to-day radar operations.

Default layout:
- one large primary pane
- optional compact right dock open on Alerts or Inspect
- time deck visible

Hot-path controls:
- station focus / auto nearest / lock
- product chips
- tilt ladder
- loop transport
- alerts badge

This is the launch workspace.

### 2. Compare workspace
For two-up and four-up interrogation.

Prebuilt presets:
- **Dual Linked**
- **2x2 Tornado**
- **2x2 Hail**
- **Before / After**
- **Custom Grid**

Each pane header needs:
- station
- product
- tilt
- time label
- link chips
- clone / swap / close

Link dimensions:
- geography
- time
- tilt
- station
- product family (optional)

### 3. Archive workspace
Archive should not feel like a different app.

Use the same main layout as Live, but the right dock defaults to:
- case library
- time range
- station/source picker
- bookmarks

The time deck becomes archive-aware instead of transforming into a separate timeline panel.

### 4. Warning workspace
This is the serious-meteorologist mode.

When an alert or detection is selected, this workspace should open a preset interrogation layout automatically.

#### Tornado preset
Recommended default panes:
- Pane A: lowest tilt reflectivity
- Pane B: lowest tilt velocity / SRV
- Pane C: CC / TDS relevant product
- Pane D: higher-tilt velocity or trend loop

Right dock shows:
- warning card
- office / issue / expiration time
- nearest/best radar recommendation
- quick actions: Focus, Interrogate, Follow, Bookmark

### 5. Volume / Cross-Section workspace
3D and cross-section should feel like purpose-built analysis modes, not button side quests.

#### 3D mode layout
- main 3D canvas full center
- small linked 2D locator inset
- compact controls for vertical exaggeration / opacity / slice options
- time deck remains active when compatible

#### Cross-section mode layout
- top: 2D map with baseline
- bottom: cross-section panel
- right dock: baseline controls, depth, source pane, interpolation options

---

## Converting current Cursdar2 features into the new homes

### Move out of the main operator console
These should no longer live in a giant left stack:
- basemap controls
- performance profile details and memory telemetry
- color table loading/reset
- polling links / placefile management
- archive download form
- historic case library
- detailed warning styling

### Keep in the top bar / hot path
These stay instantly visible:
- station
- product
- tilt
- mode
- live/loop controls
- compare layout
- auto-track / lock
- alert badge

### Put in the right dock
- selected station details
- warning list/details
- overlay toggles
- layer controls
- cursor probe details

### Put in the asset manager
- palettes
- polling links / placefiles
- archive downloader
- feed/source management
- saved layouts / workspaces

---

## Key interaction design

### Product selection
Replace vertical button stacks with a horizontal product chip row.

Behavior:
- single click = switch
- hover = tooltip / hotkey / quick description
- mouse wheel over chip row = cycle
- right click = pin into compare pane

### Tilt selection
Replace two separate up/down buttons with a real tilt ladder.

Behavior:
- labeled sweep chips or compact vertical ladder
- unavailable tilts appear dimmed
- hover shows angle + availability
- shift+scroll over ladder cycles fast

### Station selection
The station browser should become an on-demand explorer, not a permanent window.

Recommended station UX:
- top search field with ICAO/city/state fuzzy search
- recent stations + favorites
- hover card on map for nearest station
- lock/unlock active station from top bar
- optional explorer drawer for full inventory when needed

### Alerts
Warnings should be card-based, not checkbox soup + raw list.

Each alert card:
- event type
- severity color band
- office
- issue/expire time
- quick actions: focus, interrogate, follow, bookmark
- optional badges: live / historic / nearest radar / TDS nearby

### Cursor interrogation
Add a true **Probe HUD**.

At minimum show:
- lat/lon
- product value under cursor
- range / azimuth relative to active radar
- source radar / sweep / scan time
- storm-relative values when applicable

Advanced:
- Alt-click pins a probe
- linked panes compare values at equivalent location

### Context menus
Right-click on map should open a compact radial/context menu with the most likely action for that object.

Examples:
- on station: focus, lock, enable, compare, open archive
- on warning polygon: focus, interrogate, follow, bookmark
- on blank map: drop probe, create cross-section, set home, search nearest radar

---

## Saved workspace presets
Cursdar3 needs named presets. Not optional.

Starter presets:
- **Solo Live**
- **Dual Linked**
- **Tornado Interrogate**
- **Hail Interrogate**
- **Archive Review**
- **3D Inspect**
- **Cross-Section**

Each preset stores:
- pane layout
- linked dimensions
- default dock tab
- default product set
- overlay set
- time deck behavior

This is how you stop future feature clutter from returning.

---

## Visual design language

### Theme direction
Not gamer neon. Not sterile enterprise. Think:
- dark tactical glass
- restrained military/aviation console energy
- radar data colors are brightest thing on screen
- UI chrome is graphite / slate / steel blue

### Color usage
Reserve strong colors for meaning:
- green = live / ready
- amber = archive / review / pending
- red = warning / critical alert
- cyan = linked / active tool / download state
- gray = idle / unavailable / disabled

### Typography
- compact sans for labels
- optional monospace for timestamps, ICAO, scan ages, telemetry
- avoid oversized headings
- optimize for dense but calm readability

### Motion
Animations should communicate readiness, not decoration.

Rules:
- under 120 ms for panel open/close where possible
- crossfade or subtle slide only
- no long easing curves
- nothing that implies lag

---

## The biggest architectural fix: one global time model
This is the most important redesign call.

Currently, live loop and archive playback feel like separate systems.
In Cursdar3 they must become one **Temporal Controller**.

### Temporal Controller state
- source mode: live / archive / snapshot
- play state: playing / paused / live-tail
- playback cursor
- target loop span
- fps
- linked panes
- bookmarks / markers
- pending frame state

### Why this matters
If time is global:
- play can carry over across product and tilt switches
- compare panes stay synchronized
- archive and live feel like siblings instead of separate workflows
- future features like warning markers, bookmarks, exports, and evidence review become much easier

---

## The biggest organizational fix: one context dock instead of three side windows
Current separate concepts like Inspector, Station Browser, and Warnings should become views of the same contextual side dock.

Why:
- less window hunting
- less duplicated data
- selection drives the UI
- easier to scale features later

Selection examples:
- selecting a station => Inspect tab becomes station details
- selecting a warning => Alerts tab becomes detailed warning card
- selecting nothing => Inspect tab becomes cursor/context overview

---

## The biggest workflow fix: interrogation presets
A meteorologist hunting tornado evidence does not want to manually build a layout from scratch every time.

Cursdar3 should support one-click interrogation presets:
- select warning or TDS marker
- app opens Tornado Interrogate preset
- pans/zooms to target
- picks best linked products
- starts time deck in linked review mode
- preserves play state if already playing

This is a major identity feature.

---

## What to de-emphasize visually
The following are important but should not dominate the main experience:
- VRAM/RAM telemetry
- archive download forms
- palette file path controls
- polling-link setup
- rarely changed warning style controls
- verbose station inventory lists

These belong in drawers/managers, not in the operator's main line of sight.

---

## Implementation suggestion for the coding agent
Do **not** begin by rewriting the rendering engine.

### Recommended approach
Keep the current rendering/data core and rebuild the UI shell around a new view-state model.

#### New high-level state objects
```cpp
struct WorkspaceState {
    WorkspaceMode mode;          // Live, Compare, Archive, Warning, Volume
    LayoutPreset preset;
    DockTab activeDockTab;
};

struct TemporalState {
    TimeSource source;           // Live, Archive, Snapshot
    bool playing;
    bool liveTail;
    int cursorFrame;
    int loopLength;
    float fps;
};

struct PaneState {
    int stationId;
    Product product;
    int tiltIndex;
    LinkMask links;              // Geo, Time, Tilt, Station
};

struct SelectionState {
    SelectionType type;          // None, Station, Warning, Detection, Probe
    int id;
};
```

### Build order
1. Create the new workspace rail, top bar, right dock, and bottom time deck shell.
2. Move product/tilt/station hot-path controls to the top bar.
3. Unify live loop + historic timeline into one Temporal Controller.
4. Convert Inspector / Warnings / Station Browser into right-dock tabs.
5. Move archive / palettes / polling links / performance into an Asset Manager drawer.
6. Add saved presets and pane linking.
7. Add warning-triggered interrogation presets.

---

## Acceptance criteria
Cursdar3 is successful when all of these are true:

### Hot path
- A user can change station, product, tilt, and loop state without opening any drawer.
- The radar canvas remains visually dominant at all times.
- Play state carries across compatible product/tilt changes.

### Clarity
- There is no "giant everything console" anymore.
- No more than five permanent UI regions exist at once.
- Advanced settings do not crowd active analysis.

### Warning operations
- Clicking an alert can immediately open an interrogation layout.
- Compare panes can be linked by geography and time.
- The operator can chase pixel-level evidence without window juggling.

### Scalability
- New features can be added without creating a new permanent panel.
- Saved layouts/presets prevent future clutter regression.

---

## Final directive to the coding agent
Build Cursdar3 as a **workspace-driven radar workstation**, not as a collection of feature panels.

The redesign is complete only when:
- the UI sells the speed of the engine,
- the default experience is simpler than Cursdar2,
- the advanced workflows are deeper than Cursdar2,
- and a warning meteorologist can get to the right evidence layout in one or two actions.
