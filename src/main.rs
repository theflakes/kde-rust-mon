mod collector;

use collector::{HardwareCollector, SystemSnapshot};
use eframe::egui::{
    self, Color32, FontData, FontDefinitions, FontFamily, FontId, ProgressBar, RichText, TextStyle,
    Vec2, ViewportBuilder, ViewportCommand, WindowLevel,
};
use ksni::{
    Icon, MenuItem, ToolTip, Tray, TrayService,
    menu::{RadioGroup, RadioItem, StandardItem},
};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;

const POLL_INTERVALS_MS: [u64; 4] = [500, 1000, 1500, 2000];
const IPC_INTERVAL_FILE: &str = "/tmp/kde-rust-mon-interval";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FontChoice {
    Sans,
    Mono,
    Serif,
}

impl FontChoice {
    const ALL: [FontChoice; 3] = [FontChoice::Sans, FontChoice::Mono, FontChoice::Serif];

    fn label(self) -> &'static str {
        match self {
            FontChoice::Sans => "Sans",
            FontChoice::Mono => "Mono",
            FontChoice::Serif => "Serif",
        }
    }

    fn as_key(self) -> &'static str {
        match self {
            FontChoice::Sans => "sans",
            FontChoice::Mono => "mono",
            FontChoice::Serif => "serif",
        }
    }

    fn from_key(s: &str) -> Self {
        match s.trim() {
            "mono" => FontChoice::Mono,
            "serif" => FontChoice::Serif,
            _ => FontChoice::Sans,
        }
    }

    fn to_egui_family(self) -> FontFamily {
        match self {
            FontChoice::Sans => FontFamily::Proportional,
            FontChoice::Mono => FontFamily::Monospace,
            FontChoice::Serif => FontFamily::Name("Serif".into()),
        }
    }
}

#[derive(Clone, Debug)]
struct AppConfig {
    font_choice: FontChoice,
    font_size: f32,
    interval_ms: u64,
    always_on_top: bool,
    show_per_core: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            font_choice: FontChoice::Sans,
            font_size: 13.0,
            interval_ms: 1000,
            always_on_top: false,
            show_per_core: false,
        }
    }
}

impl AppConfig {
    fn config_path() -> PathBuf {
        if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join("kde-rust-mon.conf");
            }
        }
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(".config/kde-rust-mon.conf");
        }
        PathBuf::from("/tmp/kde-rust-mon.conf")
    }

    fn load() -> Self {
        let mut cfg = Self::default();
        let Ok(content) = fs::read_to_string(Self::config_path()) else {
            return cfg;
        };

        for line in content.lines() {
            if let Some((k, v)) = line.split_once('=') {
                match k.trim() {
                    "font_family" => cfg.font_choice = FontChoice::from_key(v),
                    "font_size" => {
                        if let Ok(sz) = v.trim().parse::<f32>() {
                            cfg.font_size = sz.clamp(9.0, 26.0);
                        }
                    }
                    "interval_ms" => {
                        if let Ok(ms) = v.trim().parse::<u64>() {
                            if POLL_INTERVALS_MS.contains(&ms) {
                                cfg.interval_ms = ms;
                            }
                        }
                    }
                    "always_on_top" => cfg.always_on_top = v.trim() == "true",
                    "show_per_core" => cfg.show_per_core = v.trim() == "true",
                    _ => {}
                }
            }
        }
        cfg
    }

    fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let body = format!(
            "font_family={}\nfont_size={:.1}\ninterval_ms={}\nalways_on_top={}\nshow_per_core={}\n",
            self.font_choice.as_key(),
            self.font_size,
            self.interval_ms,
            self.always_on_top,
            self.show_per_core
        );
        let _ = fs::write(path, body);
    }
}

struct TraySharedState {
    interval_ms: AtomicU64,
    gui_child: Mutex<Option<Child>>,
}

impl TraySharedState {
    fn is_window_open(&self) -> bool {
        if let Ok(mut guard) = self.gui_child.lock() {
            if let Some(child) = guard.as_mut() {
                if let Ok(None) = child.try_wait() {
                    return true;
                }
                *guard = None;
            }
        }
        false
    }

    fn open_window(&self) {
        if self.is_window_open() {
            return;
        }
        if let Ok(exe) = env::current_exe() {
            let ms = self.interval_ms.load(Ordering::Relaxed);
            let _ = fs::write(IPC_INTERVAL_FILE, ms.to_string());
            if let Ok(child) = Command::new(exe).arg("--gui-worker").spawn() {
                if let Ok(mut guard) = self.gui_child.lock() {
                    *guard = Some(child);
                }
            }
        }
    }

    fn close_window(&self) {
        if let Ok(mut guard) = self.gui_child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    fn toggle_window(&self) {
        if self.is_window_open() {
            self.close_window();
        } else {
            self.open_window();
        }
    }

    fn set_interval(&self, ms: u64) {
        self.interval_ms.store(ms, Ordering::Relaxed);
        let _ = fs::write(IPC_INTERVAL_FILE, ms.to_string());
        let mut cfg = AppConfig::load();
        cfg.interval_ms = ms;
        cfg.save();
    }

    fn sync_interval_from_ipc(&self) -> u64 {
        if let Ok(s) = fs::read_to_string(IPC_INTERVAL_FILE) {
            if let Ok(ms) = s.trim().parse::<u64>() {
                if POLL_INTERVALS_MS.contains(&ms) {
                    self.interval_ms.store(ms, Ordering::Relaxed);
                    return ms;
                }
            }
        }
        self.interval_ms.load(Ordering::Relaxed)
    }
}

struct KdeResourceTray {
    snapshot: SystemSnapshot,
    shared: Arc<TraySharedState>,
}

impl Tray for KdeResourceTray {
    fn id(&self) -> String {
        "org.kde.rustmon".into()
    }

    fn title(&self) -> String {
        let gpu_summary = self
            .snapshot
            .gpus
            .iter()
            .map(|g| format!("{}%", g.load_pct))
            .collect::<Vec<_>>()
            .join("|");

        if gpu_summary.is_empty() {
            format!(
                "CPU: {:.0}% | RAM: {:.0}%",
                self.snapshot.cpu_total_pct, self.snapshot.mem.ram_pct
            )
        } else {
            format!(
                "CPU: {:.0}% | RAM: {:.0}% | GPU: {}",
                self.snapshot.cpu_total_pct, self.snapshot.mem.ram_pct, gpu_summary
            )
        }
    }

    fn icon_name(&self) -> String {
        "utilities-system-monitor".into()
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        self.shared.toggle_window();
    }

    fn icon_pixmap(&self) -> Vec<Icon> {
        let width = 22;
        let height = 22;
        let mut data = vec![0u8; (width * height * 4) as usize];

        let cpu_fill = ((self.snapshot.cpu_total_pct / 100.0) * height as f32).round() as i32;
        draw_bar(&mut data, width, height, 1, 6, cpu_fill, [255, 0, 188, 212]);

        let ram_fill = ((self.snapshot.mem.ram_pct / 100.0) * height as f32).round() as i32;
        draw_bar(
            &mut data,
            width,
            height,
            8,
            13,
            ram_fill,
            [255, 171, 71, 188],
        );

        let max_gpu = self
            .snapshot
            .gpus
            .iter()
            .map(|g| g.load_pct)
            .max()
            .unwrap_or(0);
        let gpu_fill = ((max_gpu as f32 / 100.0) * height as f32).round() as i32;
        draw_bar(
            &mut data,
            width,
            height,
            15,
            20,
            gpu_fill,
            [255, 76, 175, 80],
        );

        vec![Icon {
            width,
            height,
            data,
        }]
    }

    fn tool_tip(&self) -> ToolTip {
        let m = &self.snapshot.mem;
        let ram_speed_tag = if m.ram_speed_mts > 0 {
            format!(" @ {} MT/s", m.ram_speed_mts)
        } else {
            String::new()
        };

        let cpu_freq_tag = if self.snapshot.cpu_avg_mhz > 0 {
            format!(" @ {}", format_mhz(self.snapshot.cpu_avg_mhz))
        } else {
            String::new()
        };

        let mut html = format!(
            "<b>CPU Total:</b> {:.1}%{} ({:.1} °C)<br/>\
             <b>RAM:</b> {:.1} / {:.1} GiB ({:.1}%){}<br/>",
            self.snapshot.cpu_total_pct,
            cpu_freq_tag,
            self.snapshot.cpu_temp_c,
            m.ram_used_mb as f32 / 1024.0,
            m.ram_total_mb as f32 / 1024.0,
            m.ram_pct,
            ram_speed_tag,
        );

        if m.swap_total_mb > 0 {
            html.push_str(&format!(
                "<b>Swap:</b> {:.2} / {:.2} GiB ({:.1}%)<br/>",
                m.swap_used_mb as f32 / 1024.0,
                m.swap_total_mb as f32 / 1024.0,
                m.swap_pct
            ));
        } else {
            html.push_str("<b>Swap:</b> Disabled<br/>");
        }

        for z in &self.snapshot.zrams {
            html.push_str(&format!(
                "<b>ZRAM ({} [{}]):</b> Data: {} → Compr: {} ({:.2}x)<br/>\
                 &nbsp;&nbsp;RAM Used: {} / Disksize: {}<br/>",
                z.name,
                z.algorithm,
                format_bytes(z.data_bytes),
                format_bytes(z.compr_bytes),
                z.ratio,
                format_bytes(z.total_ram_bytes),
                format_bytes(z.disksize_bytes),
            ));
        }

        if !self.snapshot.gpus.is_empty() {
            html.push_str("<hr/>");
            for (idx, gpu) in self.snapshot.gpus.iter().enumerate() {
                let mem_line = if gpu.vram_total_mb > 0 {
                    let vclk = if gpu.vram_freq_mhz > 0 {
                        format!(" @ {} MHz", gpu.vram_freq_mhz)
                    } else {
                        String::new()
                    };
                    format!(
                        "VRAM: {} / {} MiB{}",
                        gpu.vram_used_mb, gpu.vram_total_mb, vclk
                    )
                } else if m.ram_speed_mts > 0 {
                    format!("Unified Memory @ {} MT/s", m.ram_speed_mts)
                } else {
                    "Unified Memory (Shared System RAM)".into()
                };

                html.push_str(&format!(
                    "<b>GPU {idx} ({} [{}]):</b> {}% ({}/{} MHz) | {} °C | {:.1} W<br/>{}<br/>",
                    gpu.name,
                    gpu.id,
                    gpu.load_pct,
                    gpu.freq_mhz,
                    gpu.max_freq_mhz,
                    gpu.temp_c,
                    gpu.power_w,
                    mem_line
                ));
            }
        }

        ToolTip {
            icon_name: "utilities-system-monitor".into(),
            icon_pixmap: Vec::new(),
            title: "KDE Rust Hardware Monitor (Click to Toggle Window)".into(),
            description: html,
        }
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let shared_toggle = self.shared.clone();
        let shared_radio = self.shared.clone();
        let shared_quit = self.shared.clone();

        let current_ms = self.shared.interval_ms.load(Ordering::Relaxed);
        let selected_idx = POLL_INTERVALS_MS
            .iter()
            .position(|&ms| ms == current_ms)
            .unwrap_or(1);

        vec![
            StandardItem {
                label: "Toggle Stats Window".into(),
                icon_name: "utilities-system-monitor".into(),
                activate: Box::new(move |_| {
                    shared_toggle.toggle_window();
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            RadioGroup {
                selected: selected_idx,
                select: Box::new(move |_, idx| {
                    if let Some(&ms) = POLL_INTERVALS_MS.get(idx) {
                        shared_radio.set_interval(ms);
                    }
                }),
                options: vec![
                    RadioItem {
                        label: "Poll: 0.5s (500ms)".into(),
                        ..Default::default()
                    },
                    RadioItem {
                        label: "Poll: 1.0s (1000ms)".into(),
                        ..Default::default()
                    },
                    RadioItem {
                        label: "Poll: 1.5s (1500ms)".into(),
                        ..Default::default()
                    },
                    RadioItem {
                        label: "Poll: 2.0s (2000ms)".into(),
                        ..Default::default()
                    },
                ],
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".into(),
                icon_name: "application-exit".into(),
                activate: Box::new(move |_| {
                    shared_quit.close_window();
                    std::process::exit(0);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

// ============================================================================
// GUI Worker Window Process
// ============================================================================

struct GuiWorkerState {
    snapshot: Mutex<SystemSnapshot>,
    interval_ms: AtomicU64,
    egui_ctx: Mutex<Option<egui::Context>>,
}

struct MonitorWindowApp {
    state: Arc<GuiWorkerState>,
    config: AppConfig,
    /// When > 0, fits the window tightly to the rendered content on the next couple layout passes
    fit_passes_remaining: u8,
    last_fitted_size: Vec2,
}

impl MonitorWindowApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        state: Arc<GuiWorkerState>,
        config: AppConfig,
    ) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        setup_system_fonts(&cc.egui_ctx);
        apply_font_style(&cc.egui_ctx, config.font_choice, config.font_size);

        if config.always_on_top {
            cc.egui_ctx
                .send_viewport_cmd(ViewportCommand::WindowLevel(WindowLevel::AlwaysOnTop));
        }

        if let Ok(mut guard) = state.egui_ctx.lock() {
            *guard = Some(cc.egui_ctx.clone());
        }

        Self {
            state,
            config,
            fit_passes_remaining: 3,
            last_fitted_size: Vec2::ZERO,
        }
    }

    fn trigger_auto_fit(&mut self) {
        self.fit_passes_remaining = 3;
    }
}

/// Loads available system TTF fonts (Sans, Mono, Serif) so the user can switch families live.
fn setup_system_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();

    let sans_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ];
    let mono_candidates = [
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ];
    let serif_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    ];

    if let Some(bytes) = sans_candidates.iter().find_map(|p| fs::read(p).ok()) {
        fonts
            .font_data
            .insert("sys_sans".to_owned(), FontData::from_owned(bytes));
        if let Some(vec) = fonts.families.get_mut(&FontFamily::Proportional) {
            vec.insert(0, "sys_sans".to_owned());
        }
    }

    if let Some(bytes) = mono_candidates.iter().find_map(|p| fs::read(p).ok()) {
        fonts
            .font_data
            .insert("sys_mono".to_owned(), FontData::from_owned(bytes));
        if let Some(vec) = fonts.families.get_mut(&FontFamily::Monospace) {
            vec.insert(0, "sys_mono".to_owned());
        }
    }

    let mut serif_list = Vec::new();
    if let Some(bytes) = serif_candidates.iter().find_map(|p| fs::read(p).ok()) {
        fonts
            .font_data
            .insert("sys_serif".to_owned(), FontData::from_owned(bytes));
        serif_list.push("sys_serif".to_owned());
    }
    // Fallback to proportional if no serif TTF is installed
    if let Some(prop) = fonts.families.get(&FontFamily::Proportional) {
        serif_list.extend(prop.iter().cloned());
    }
    fonts
        .families
        .insert(FontFamily::Name("Serif".into()), serif_list);

    ctx.set_fonts(fonts);
}

fn apply_font_style(ctx: &egui::Context, choice: FontChoice, base_size: f32) {
    let mut style = (*ctx.style()).clone();
    let family = choice.to_egui_family();

    let mut text_styles = BTreeMap::new();
    text_styles.insert(
        TextStyle::Small,
        FontId::new((base_size * 0.85).max(8.0), family.clone()),
    );
    text_styles.insert(TextStyle::Body, FontId::new(base_size, family.clone()));
    text_styles.insert(TextStyle::Button, FontId::new(base_size, family.clone()));
    text_styles.insert(
        TextStyle::Monospace,
        FontId::new(base_size, FontFamily::Monospace),
    );
    text_styles.insert(TextStyle::Heading, FontId::new(base_size * 1.18, family));

    style.text_styles = text_styles;
    ctx.set_style(style);
}

impl eframe::App for MonitorWindowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let snap = self
            .state
            .snapshot
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();

        let mut cur_interval = self.state.interval_ms.load(Ordering::Relaxed);
        let mut config_dirty = false;
        let mut layout_changed = false;

        let top_resp = egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("KDE Monitor").strong());

                ui.separator();

                // Font Family selector
                let prev_font = self.config.font_choice;
                egui::ComboBox::from_id_salt("font_family_combo")
                    .selected_text(self.config.font_choice.label())
                    .width(56.0)
                    .show_ui(ui, |ui| {
                        for choice in FontChoice::ALL {
                            ui.selectable_value(
                                &mut self.config.font_choice,
                                choice,
                                choice.label(),
                            );
                        }
                    });
                if self.config.font_choice != prev_font {
                    apply_font_style(ctx, self.config.font_choice, self.config.font_size);
                    config_dirty = true;
                    layout_changed = true;
                }

                // Font Size [-] [13] [+] controls
                if ui
                    .small_button("−")
                    .on_hover_text("Decrease font size")
                    .clicked()
                {
                    self.config.font_size = (self.config.font_size - 1.0).max(9.0);
                    apply_font_style(ctx, self.config.font_choice, self.config.font_size);
                    config_dirty = true;
                    layout_changed = true;
                }
                ui.label(format!("{:.0}pt", self.config.font_size));
                if ui
                    .small_button("+")
                    .on_hover_text("Increase font size")
                    .clicked()
                {
                    self.config.font_size = (self.config.font_size + 1.0).min(24.0);
                    apply_font_style(ctx, self.config.font_choice, self.config.font_size);
                    config_dirty = true;
                    layout_changed = true;
                }

                ui.separator();

                // Refresh interval selector
                egui::ComboBox::from_id_salt("poll_rate_combo")
                    .selected_text(format!("{:.1}s", cur_interval as f32 / 1000.0))
                    .width(52.0)
                    .show_ui(ui, |ui| {
                        for &ms in &POLL_INTERVALS_MS {
                            let label = format!("{:.1}s", ms as f32 / 1000.0);
                            if ui.selectable_value(&mut cur_interval, ms, label).changed() {
                                self.state
                                    .interval_ms
                                    .store(cur_interval, Ordering::Relaxed);
                                let _ = fs::write(IPC_INTERVAL_FILE, cur_interval.to_string());
                                self.config.interval_ms = cur_interval;
                                config_dirty = true;
                            }
                        }
                    });

                if ui.checkbox(&mut self.config.always_on_top, "Pin").changed() {
                    let level = if self.config.always_on_top {
                        WindowLevel::AlwaysOnTop
                    } else {
                        WindowLevel::Normal
                    };
                    ctx.send_viewport_cmd(ViewportCommand::WindowLevel(level));
                    config_dirty = true;
                }

                if ui
                    .small_button("Fit")
                    .on_hover_text("Shrink window to fit content")
                    .clicked()
                {
                    layout_changed = true;
                }

                if ui.button("Hide").clicked() {
                    ctx.send_viewport_cmd(ViewportCommand::Close);
                }
            })
            .response
        });

        let top_bar_req_w = top_resp.inner.rect.width();
        let top_bar_h = top_resp.response.rect.height();

        let mut content_size = Vec2::ZERO;

        egui::CentralPanel::default().show(ctx, |ui| {
            // Measure exact natural size of the stacked metric cards without expanding
            let inner = ui.vertical(|ui| {
                // --- CPU SECTION ---
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("CPU").strong());
                        ui.label(
                            RichText::new(format!("{:.1} °C", snap.cpu_temp_c))
                                .color(Color32::from_rgb(0, 188, 212)),
                        );
                        if snap.cpu_avg_mhz > 0 {
                            ui.label(
                                RichText::new(format!("• Avg: {}", format_mhz(snap.cpu_avg_mhz)))
                                    .color(Color32::from_rgb(128, 222, 234)),
                            );
                        }
                        if ui
                            .checkbox(&mut self.config.show_per_core, "Per-Core")
                            .changed()
                        {
                            config_dirty = true;
                            layout_changed = true;
                        }
                    });

                    let total_label = if snap.cpu_avg_mhz > 0 {
                        format!(
                            "Total: {:.1}%  (Avg: {})",
                            snap.cpu_total_pct,
                            format_mhz(snap.cpu_avg_mhz)
                        )
                    } else {
                        format!("Total: {:.1}%", snap.cpu_total_pct)
                    };

                    ui.add(
                        ProgressBar::new(snap.cpu_total_pct / 100.0)
                            .text(total_label)
                            .fill(Color32::from_rgb(0, 150, 180)),
                    );

                    if self.config.show_per_core && !snap.cpu_cores_pct.is_empty() {
                        ui.add_space(4.0);
                        let cols = if snap.cpu_cores_mhz.is_empty() { 4 } else { 3 };
                        // Scale cell width proportionally with the user's chosen font size
                        let scale = self.config.font_size / 13.0;
                        let cell_w = if snap.cpu_cores_mhz.is_empty() {
                            88.0 * scale
                        } else {
                            126.0 * scale
                        };

                        egui::Grid::new("cpu_cores_grid")
                            .num_columns(cols)
                            .spacing([6.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                for (idx, &pct) in snap.cpu_cores_pct.iter().enumerate() {
                                    let cell_text = if let Some(&mhz) = snap.cpu_cores_mhz.get(idx)
                                    {
                                        if mhz > 0 {
                                            format!(
                                                "C{idx}: {pct:.0}% • {:.2}G",
                                                mhz as f32 / 1000.0
                                            )
                                        } else {
                                            format!("C{idx}: {pct:.0}%")
                                        }
                                    } else {
                                        format!("C{idx}: {pct:.0}%")
                                    };

                                    ui.add(
                                        ProgressBar::new(pct / 100.0)
                                            .text(cell_text)
                                            .desired_width(cell_w),
                                    );
                                    if (idx + 1) % cols == 0 {
                                        ui.end_row();
                                    }
                                }
                            });
                    }
                });

                ui.add_space(4.0);

                // --- MEMORY & ZRAM SECTION ---
                ui.group(|ui| {
                    let m = &snap.mem;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Memory & Swap").strong());
                        if m.ram_speed_mts > 0 {
                            ui.label(
                                RichText::new(format!("• {} MT/s", m.ram_speed_mts))
                                    .color(Color32::from_rgb(186, 104, 200)),
                            );
                        }
                    });

                    ui.add(
                        ProgressBar::new(m.ram_pct / 100.0)
                            .text(format!(
                                "RAM: {:.2} / {:.2} GiB ({:.1}%)",
                                m.ram_used_mb as f32 / 1024.0,
                                m.ram_total_mb as f32 / 1024.0,
                                m.ram_pct
                            ))
                            .fill(Color32::from_rgb(142, 68, 173)),
                    );

                    if m.swap_total_mb > 0 {
                        ui.add_space(2.0);
                        ui.add(
                            ProgressBar::new(m.swap_pct / 100.0)
                                .text(format!(
                                    "Swap: {:.2} / {:.2} GiB ({:.1}%)",
                                    m.swap_used_mb as f32 / 1024.0,
                                    m.swap_total_mb as f32 / 1024.0,
                                    m.swap_pct
                                ))
                                .fill(Color32::from_rgb(110, 80, 150)),
                        );
                    }

                    for z in &snap.zrams {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("ZRAM ({} [{}])", z.name, z.algorithm))
                                    .strong(),
                            );
                            ui.label(
                                RichText::new(format!("Ratio: {:.2}x", z.ratio))
                                    .color(Color32::from_rgb(129, 199, 132)),
                            );
                        });
                        ui.label(format!(
                            "Data: {} → Compr: {} | RAM: {} / {}",
                            format_bytes(z.data_bytes),
                            format_bytes(z.compr_bytes),
                            format_bytes(z.total_ram_bytes),
                            format_bytes(z.disksize_bytes),
                        ));
                    }
                });

                // --- GPUS SECTION ---
                for (idx, gpu) in snap.gpus.iter().enumerate() {
                    ui.add_space(4.0);
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("GPU {idx}: {} [{}]", gpu.name, gpu.id))
                                    .strong(),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "• {} °C | {:.1} W",
                                    gpu.temp_c, gpu.power_w
                                ))
                                .color(Color32::from_rgb(102, 187, 106)),
                            );
                        });

                        ui.add(
                            ProgressBar::new(gpu.load_pct as f32 / 100.0)
                                .text(format!(
                                    "Clock Load: {}%  ({} / {} MHz)",
                                    gpu.load_pct, gpu.freq_mhz, gpu.max_freq_mhz
                                ))
                                .fill(Color32::from_rgb(56, 142, 60)),
                        );

                        if gpu.vram_total_mb > 0 {
                            let vram_frac = (gpu.vram_used_mb as f32 / gpu.vram_total_mb as f32)
                                .clamp(0.0, 1.0);
                            let vclk_str = if gpu.vram_freq_mhz > 0 {
                                format!(" @ {} MHz", gpu.vram_freq_mhz)
                            } else {
                                String::new()
                            };
                            ui.add_space(2.0);
                            ui.add(
                                ProgressBar::new(vram_frac)
                                    .text(format!(
                                        "VRAM: {} / {} MiB ({:.0}%){}",
                                        gpu.vram_used_mb,
                                        gpu.vram_total_mb,
                                        vram_frac * 100.0,
                                        vclk_str
                                    ))
                                    .fill(Color32::from_rgb(46, 110, 90)),
                            );
                        } else {
                            let unified_text = if snap.mem.ram_speed_mts > 0 {
                                format!(
                                    "Unified Memory (Shared RAM @ {} MT/s)",
                                    snap.mem.ram_speed_mts
                                )
                            } else {
                                "Unified Memory (Shared System RAM)".to_string()
                            };
                            ui.label(RichText::new(unified_text).weak());
                        }
                    });
                }
            });

            content_size = inner.response.rect.size();
        });

        if config_dirty {
            self.config.save();
        }
        if layout_changed {
            self.trigger_auto_fit();
        }

        // Fit window tightly to the displayed text & bars without locking out manual resizing
        if self.fit_passes_remaining > 0 {
            self.fit_passes_remaining -= 1;
            let margin = ctx.style().spacing.window_margin;
            let target_w = content_size.x.max(top_bar_req_w) + margin.left + margin.right + 4.0;
            let target_h = top_bar_h + content_size.y + margin.top + margin.bottom + 4.0;
            let target = Vec2::new(target_w.ceil(), target_h.ceil());

            if target.x > 50.0 && target.y > 50.0 && (target - self.last_fitted_size).length() > 1.0
            {
                self.last_fitted_size = target;
                ctx.send_viewport_cmd(ViewportCommand::InnerSize(target));
            }
        }
    }
}

fn run_gui_worker() -> anyhow::Result<()> {
    let mut collector = HardwareCollector::new()?;
    let initial_snapshot = collector.sample()?;
    let config = AppConfig::load();

    // Sync IPC interval with saved config on launch
    let initial_ms = fs::read_to_string(IPC_INTERVAL_FILE)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|ms| POLL_INTERVALS_MS.contains(ms))
        .unwrap_or(config.interval_ms);
    let _ = fs::write(IPC_INTERVAL_FILE, initial_ms.to_string());

    let state = Arc::new(GuiWorkerState {
        snapshot: Mutex::new(initial_snapshot),
        interval_ms: AtomicU64::new(initial_ms),
        egui_ctx: Mutex::new(None),
    });

    let state_bg = state.clone();
    thread::spawn(move || {
        loop {
            let sleep_ms = fs::read_to_string(IPC_INTERVAL_FILE)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .filter(|ms| POLL_INTERVALS_MS.contains(ms))
                .unwrap_or_else(|| state_bg.interval_ms.load(Ordering::Relaxed));
            state_bg.interval_ms.store(sleep_ms, Ordering::Relaxed);

            thread::sleep(Duration::from_millis(sleep_ms));

            if let Ok(snap) = collector.sample() {
                if let Ok(mut guard) = state_bg.snapshot.lock() {
                    *guard = snap;
                }
                if let Ok(guard) = state_bg.egui_ctx.lock() {
                    if let Some(ctx) = &*guard {
                        ctx.request_repaint();
                    }
                }
            }
        }
    });

    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title("KDE Rust Hardware Monitor")
            .with_app_id("org.kde.rustmon")
            .with_resizable(true)
            .with_inner_size([360.0, 220.0])
            .with_min_inner_size([260.0, 140.0]),
        ..Default::default()
    };

    eframe::run_native(
        "KDE Rust Hardware Monitor",
        native_options,
        Box::new(move |cc| Ok(Box::new(MonitorWindowApp::new(cc, state, config)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}

fn format_mhz(mhz: u32) -> String {
    if mhz >= 1000 {
        format!("{:.2} GHz", mhz as f32 / 1000.0)
    } else {
        format!("{mhz} MHz")
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1_073_741_824.0;
    const MIB: f64 = 1_048_576.0;
    const KIB: f64 = 1_024.0;

    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1}G", b / GIB)
    } else if b >= MIB {
        format!("{:.1}M", b / MIB)
    } else if b >= KIB {
        format!("{:.1}K", b / KIB)
    } else {
        format!("{bytes}B")
    }
}

fn draw_bar(
    buf: &mut [u8],
    width: i32,
    height: i32,
    x_start: i32,
    x_end: i32,
    fill_h: i32,
    argb: [u8; 4],
) {
    for y in 0..height {
        let is_filled = (height - 1 - y) < fill_h;
        let color = if is_filled { argb } else { [80, 60, 60, 60] };
        for x in x_start..=x_end {
            let idx = ((y * width + x) * 4) as usize;
            buf[idx..idx + 4].copy_from_slice(&color);
        }
    }
}

fn main() -> anyhow::Result<()> {
    if env::args().any(|a| a == "--gui-worker") {
        return run_gui_worker();
    }

    let mut collector = HardwareCollector::new()?;
    let initial_snapshot = collector.sample()?;
    let saved_cfg = AppConfig::load();

    let start_hidden = env::args().any(|a| a == "--tray" || a == "--hidden");
    let _ = fs::write(IPC_INTERVAL_FILE, saved_cfg.interval_ms.to_string());

    let shared = Arc::new(TraySharedState {
        interval_ms: AtomicU64::new(saved_cfg.interval_ms),
        gui_child: Mutex::new(None),
    });

    if !start_hidden {
        shared.open_window();
    }

    let tray = KdeResourceTray {
        snapshot: initial_snapshot,
        shared: shared.clone(),
    };

    let service = TrayService::new(tray);
    let tray_handle = service.handle();
    service.spawn();

    loop {
        let sleep_ms = shared.sync_interval_from_ipc();
        thread::sleep(Duration::from_millis(sleep_ms));

        let _ = shared.is_window_open();

        if let Ok(snap) = collector.sample() {
            tray_handle.update(|tray: &mut KdeResourceTray| {
                tray.snapshot = snap;
            });
        }
    }
}
