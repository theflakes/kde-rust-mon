#![forbid(unsafe_code)]

mod collector;

use collector::{GpuMetrics, HardwareCollector, MemoryMetrics, SystemSnapshot, ZramMetrics};
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
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;

const POLL_INTERVALS_MS: [u64; 4] = [500, 1000, 1500, 2000];
const CTRL_SET_INTERVAL: u8 = 1;
const CTRL_SET_PER_CORE: u8 = 2;

// ============================================================================
// Zero-Dependency Binary Snapshot Framing over Standard Pipes
// ============================================================================

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_str<R: Read>(r: &mut R) -> io::Result<String> {
    let mut len_b = [0u8; 4];
    r.read_exact(&mut len_b)?;
    let len = u32::from_le_bytes(len_b) as usize;
    if len > 4096 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "string too long",
        ));
    }
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn send_gui_control_msg(tag: u8, payload: u64) {
    let mut pkt = [0u8; 9];
    pkt[0] = tag;
    pkt[1..9].copy_from_slice(&payload.to_le_bytes());
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(&pkt);
    let _ = stdout.flush();
}

fn encode_snapshot(snap: &SystemSnapshot, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&[0u8; 4]);

    out.extend_from_slice(&snap.sys_power_w.to_le_bytes());
    out.push(u8::from(snap.sys_power_is_battery));
    match snap.battery_pct {
        Some(pct) => {
            out.push(1);
            out.push(pct);
        }
        None => {
            out.push(0);
            out.push(0);
        }
    }

    out.extend_from_slice(&snap.cpu_total_pct.to_le_bytes());
    out.extend_from_slice(&(snap.cpu_cores_pct.len() as u32).to_le_bytes());
    for &p in &snap.cpu_cores_pct {
        out.extend_from_slice(&p.to_le_bytes());
    }

    out.extend_from_slice(&snap.cpu_avg_mhz.to_le_bytes());
    out.extend_from_slice(&(snap.cpu_cores_mhz.len() as u32).to_le_bytes());
    for &m in &snap.cpu_cores_mhz {
        out.extend_from_slice(&m.to_le_bytes());
    }

    out.extend_from_slice(&snap.cpu_temp_c.to_le_bytes());
    out.extend_from_slice(&snap.cpu_power_w.to_le_bytes());

    let m = &snap.mem;
    out.extend_from_slice(&m.ram_used_mb.to_le_bytes());
    out.extend_from_slice(&m.ram_total_mb.to_le_bytes());
    out.extend_from_slice(&m.ram_pct.to_le_bytes());
    out.extend_from_slice(&m.ram_speed_mts.to_le_bytes());
    out.extend_from_slice(&m.ram_current_mts.to_le_bytes());
    out.extend_from_slice(&m.ram_power_w.to_le_bytes());
    out.extend_from_slice(&m.swap_used_mb.to_le_bytes());
    out.extend_from_slice(&m.swap_total_mb.to_le_bytes());
    out.extend_from_slice(&m.swap_pct.to_le_bytes());

    out.extend_from_slice(&(snap.zrams.len() as u32).to_le_bytes());
    for z in &snap.zrams {
        write_str(out, &z.name);
        write_str(out, &z.algorithm);
        out.extend_from_slice(&z.disksize_bytes.to_le_bytes());
        out.extend_from_slice(&z.data_bytes.to_le_bytes());
        out.extend_from_slice(&z.compr_bytes.to_le_bytes());
        out.extend_from_slice(&z.total_ram_bytes.to_le_bytes());
        out.extend_from_slice(&z.ratio.to_le_bytes());
    }

    out.extend_from_slice(&(snap.gpus.len() as u32).to_le_bytes());
    for g in &snap.gpus {
        write_str(out, &g.id);
        write_str(out, &g.name);
        out.push(u8::from(g.is_integrated));
        out.extend_from_slice(&g.load_pct.to_le_bytes());
        out.extend_from_slice(&g.vram_used_mb.to_le_bytes());
        out.extend_from_slice(&g.vram_total_mb.to_le_bytes());
        out.extend_from_slice(&g.vram_freq_mhz.to_le_bytes());
        out.extend_from_slice(&g.vram_max_freq_mhz.to_le_bytes());
        out.extend_from_slice(&g.temp_c.to_le_bytes());
        out.extend_from_slice(&g.power_w.to_le_bytes());
        out.extend_from_slice(&g.freq_mhz.to_le_bytes());
        out.extend_from_slice(&g.max_freq_mhz.to_le_bytes());
    }

    let payload_len = (out.len() - 4) as u32;
    out[0..4].copy_from_slice(&payload_len.to_le_bytes());
}

fn decode_snapshot<R: Read>(r: &mut R) -> io::Result<SystemSnapshot> {
    let _payload_len = read_u32(r)?;

    let sys_power_w = read_f32(r)?;
    let sys_power_is_battery = read_u8(r)? != 0;
    let has_bat = read_u8(r)? != 0;
    let bat_val = read_u8(r)?;
    let battery_pct = if has_bat { Some(bat_val) } else { None };

    let cpu_total_pct = read_f32(r)?;
    let n_cores = read_u32(r)? as usize;
    if n_cores > 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid core count",
        ));
    }
    let mut cpu_cores_pct = Vec::with_capacity(n_cores);
    for _ in 0..n_cores {
        cpu_cores_pct.push(read_f32(r)?);
    }

    let cpu_avg_mhz = read_u32(r)?;
    let n_freqs = read_u32(r)? as usize;
    if n_freqs > 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid freq count",
        ));
    }
    let mut cpu_cores_mhz = Vec::with_capacity(n_freqs);
    for _ in 0..n_freqs {
        cpu_cores_mhz.push(read_u32(r)?);
    }

    let cpu_temp_c = read_f32(r)?;
    let cpu_power_w = read_f32(r)?;

    let mem = MemoryMetrics {
        ram_used_mb: read_u64(r)?,
        ram_total_mb: read_u64(r)?,
        ram_pct: read_f32(r)?,
        ram_speed_mts: read_u32(r)?,
        ram_current_mts: read_u32(r)?,
        ram_power_w: read_f32(r)?,
        swap_used_mb: read_u64(r)?,
        swap_total_mb: read_u64(r)?,
        swap_pct: read_f32(r)?,
    };

    let n_zrams = read_u32(r)? as usize;
    if n_zrams > 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid zram count",
        ));
    }
    let mut zrams = Vec::with_capacity(n_zrams);
    for _ in 0..n_zrams {
        zrams.push(ZramMetrics {
            name: read_str(r)?,
            algorithm: read_str(r)?,
            disksize_bytes: read_u64(r)?,
            data_bytes: read_u64(r)?,
            compr_bytes: read_u64(r)?,
            total_ram_bytes: read_u64(r)?,
            ratio: read_f32(r)?,
        });
    }

    let n_gpus = read_u32(r)? as usize;
    if n_gpus > 32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid gpu count",
        ));
    }
    let mut gpus = Vec::with_capacity(n_gpus);
    for _ in 0..n_gpus {
        gpus.push(GpuMetrics {
            id: read_str(r)?,
            name: read_str(r)?,
            is_integrated: read_u8(r)? != 0,
            load_pct: read_u32(r)?,
            vram_used_mb: read_u64(r)?,
            vram_total_mb: read_u64(r)?,
            vram_freq_mhz: read_u32(r)?,
            vram_max_freq_mhz: read_u32(r)?,
            temp_c: read_u32(r)?,
            power_w: read_f32(r)?,
            freq_mhz: read_u32(r)?,
            max_freq_mhz: read_u32(r)?,
        });
    }

    Ok(SystemSnapshot {
        sys_power_w,
        sys_power_is_battery,
        battery_pct,
        cpu_total_pct,
        cpu_cores_pct,
        cpu_avg_mhz,
        cpu_cores_mhz,
        cpu_temp_c,
        cpu_power_w,
        mem,
        zrams,
        gpus,
    })
}

// ============================================================================
// Configuration Persistence
// ============================================================================

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

// ============================================================================
// Tray Process Shared State & Dirty Tracking
// ============================================================================

struct ActiveGuiClient {
    child: Child,
    stdin: ChildStdin,
}

struct TraySharedState {
    interval_ms: AtomicU64,
    window_seq: AtomicU64,
    collect_per_core_freqs: AtomicBool,
    latest_snapshot: Mutex<SystemSnapshot>,
    gui_client: Mutex<Option<ActiveGuiClient>>,
    encode_buf: Mutex<Vec<u8>>,
}

impl TraySharedState {
    fn is_window_open(&self) -> bool {
        if let Ok(mut guard) = self.gui_client.lock() {
            if let Some(client) = guard.as_mut() {
                if let Ok(None) = client.child.try_wait() {
                    return true;
                }
                *guard = None;
            }
        }
        false
    }

    fn open_window(self: &Arc<Self>) {
        if self.is_window_open() {
            return;
        }
        let cfg = AppConfig::load();
        self.collect_per_core_freqs
            .store(cfg.show_per_core, Ordering::Relaxed);

        let Ok(exe) = env::current_exe() else {
            return;
        };

        let seq = self.window_seq.fetch_add(1, Ordering::Relaxed);

        let Ok(mut child) = Command::new(exe)
            .arg("--gui-worker")
            .arg("--piped")
            .arg(format!("--window-seq={seq}"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        else {
            return;
        };

        let Some(mut stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return;
        };

        if let (Ok(snap), Ok(mut buf)) = (self.latest_snapshot.lock(), self.encode_buf.lock()) {
            encode_snapshot(&snap, &mut buf);
            if stdin.write_all(&buf).is_err() || stdin.flush().is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
        }

        if let Some(mut stdout) = child.stdout.take() {
            let shared_ref = Arc::clone(self);
            thread::spawn(move || {
                let mut pkt = [0u8; 9];
                while stdout.read_exact(&mut pkt).is_ok() {
                    let tag = pkt[0];
                    let val = u64::from_le_bytes(pkt[1..9].try_into().unwrap_or([0; 8]));
                    match tag {
                        CTRL_SET_INTERVAL if POLL_INTERVALS_MS.contains(&val) => {
                            shared_ref.interval_ms.store(val, Ordering::Relaxed);
                        }
                        CTRL_SET_PER_CORE => {
                            shared_ref
                                .collect_per_core_freqs
                                .store(val != 0, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
                shared_ref
                    .collect_per_core_freqs
                    .store(false, Ordering::Relaxed);
            });
        }

        if let Ok(mut guard) = self.gui_client.lock() {
            *guard = Some(ActiveGuiClient { child, stdin });
        }
    }

    fn close_window(&self) {
        self.collect_per_core_freqs.store(false, Ordering::Relaxed);
        if let Ok(mut guard) = self.gui_client.lock() {
            if let Some(mut client) = guard.take() {
                drop(client.stdin);
                let _ = client.child.kill();
                let _ = client.child.wait();
            }
        }
    }

    fn toggle_window(self: &Arc<Self>) {
        if self.is_window_open() {
            self.close_window();
        } else {
            self.open_window();
        }
    }

    fn set_interval(&self, ms: u64) {
        self.interval_ms.store(ms, Ordering::Relaxed);
        let mut cfg = AppConfig::load();
        cfg.interval_ms = ms;
        cfg.save();
    }

    fn push_snapshot_to_gui(&self, snap: &SystemSnapshot) {
        let Ok(mut guard) = self.gui_client.lock() else {
            return;
        };
        let Some(client) = guard.as_mut() else {
            return;
        };

        if let Ok(Some(_)) = client.child.try_wait() {
            self.collect_per_core_freqs.store(false, Ordering::Relaxed);
            *guard = None;
            return;
        }

        if let Ok(mut buf) = self.encode_buf.lock() {
            encode_snapshot(snap, &mut buf);
            if client.stdin.write_all(&buf).is_err() || client.stdin.flush().is_err() {
                self.collect_per_core_freqs.store(false, Ordering::Relaxed);
                let _ = client.child.kill();
                let _ = client.child.wait();
                *guard = None;
            }
        }
    }
}

#[derive(PartialEq, Eq, Default)]
struct VisualDigest {
    sys_power_tenths: u32,
    battery_pct: Option<u8>,
    cpu_pct_tenths: u32,
    cpu_avg_mhz: u32,
    cpu_temp_tenths: u32,
    cpu_power_tenths: u32,
    ram_used_mb: u64,
    ram_current_mts: u32,
    ram_power_tenths: u32,
    swap_used_mb: u64,
    zram_compr_kb: u64,
    gpu_digest: Vec<(u32, u32, u32, u64, u32, u32)>,
}

impl VisualDigest {
    fn from_snapshot(s: &SystemSnapshot) -> Self {
        Self {
            sys_power_tenths: (s.sys_power_w * 10.0).round() as u32,
            battery_pct: s.battery_pct,
            cpu_pct_tenths: (s.cpu_total_pct * 10.0).round() as u32,
            cpu_avg_mhz: s.cpu_avg_mhz,
            cpu_temp_tenths: (s.cpu_temp_c * 10.0).round() as u32,
            cpu_power_tenths: (s.cpu_power_w * 10.0).round() as u32,
            ram_used_mb: s.mem.ram_used_mb,
            ram_current_mts: s.mem.ram_current_mts,
            ram_power_tenths: (s.mem.ram_power_w * 10.0).round() as u32,
            swap_used_mb: s.mem.swap_used_mb,
            zram_compr_kb: s.zrams.iter().map(|z| z.compr_bytes / 1024).sum(),
            gpu_digest: s
                .gpus
                .iter()
                .map(|g| {
                    (
                        g.load_pct,
                        g.freq_mhz,
                        g.vram_freq_mhz,
                        g.vram_used_mb,
                        g.temp_c,
                        (g.power_w * 10.0).round() as u32,
                    )
                })
                .collect(),
        }
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
        let ram_speed_tag = if m.ram_current_mts > 0 && m.ram_speed_mts > 0 {
            format!(" @ {} / {} MT/s", m.ram_current_mts, m.ram_speed_mts)
        } else if m.ram_current_mts > 0 {
            format!(" @ {} MT/s", m.ram_current_mts)
        } else if m.ram_speed_mts > 0 {
            format!(" @ {} MT/s", m.ram_speed_mts)
        } else {
            String::new()
        };

        let ram_power_tag = if m.ram_power_w > 0.0 {
            format!(" | {:.1} W", m.ram_power_w)
        } else {
            String::new()
        };

        let cpu_freq_tag = if self.snapshot.cpu_avg_mhz > 0 {
            format!(" @ {}", format_mhz(self.snapshot.cpu_avg_mhz))
        } else {
            String::new()
        };

        let cpu_power_tag = if self.snapshot.cpu_power_w > 0.0 {
            format!(" | {:.1} W", self.snapshot.cpu_power_w)
        } else {
            String::new()
        };

        let mut html = String::new();
        if self.snapshot.sys_power_w > 0.0 || self.snapshot.battery_pct.is_some() {
            let src_label = if self.snapshot.sys_power_is_battery {
                "Battery"
            } else {
                "System"
            };
            let bat_tag = self
                .snapshot
                .battery_pct
                .map(|b| format!(" (Bat: {b}%)"))
                .unwrap_or_default();
            if self.snapshot.sys_power_w > 0.0 {
                html.push_str(&format!(
                    "<b>{src_label} Power:</b> {:.1} W{bat_tag}<br/>",
                    self.snapshot.sys_power_w
                ));
            } else if let Some(b) = self.snapshot.battery_pct {
                html.push_str(&format!("<b>Battery:</b> {b}% (AC)<br/>"));
            }
        }

        html.push_str(&format!(
            "<b>CPU Total:</b> {:.1}%{} ({:.1} °C{})<br/>\
             <b>RAM:</b> {:.1} / {:.1} GiB ({:.1}%){}{}<br/>",
            self.snapshot.cpu_total_pct,
            cpu_freq_tag,
            self.snapshot.cpu_temp_c,
            cpu_power_tag,
            m.ram_used_mb as f32 / 1024.0,
            m.ram_total_mb as f32 / 1024.0,
            m.ram_pct,
            ram_speed_tag,
            ram_power_tag,
        ));

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
                "<b>ZRAM ({} [{}]):</b> Data: {} -> Compr: {} ({:.2}x)<br/>\
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
                let mem_line = if !gpu.is_integrated && gpu.vram_total_mb > 0 {
                    let vclk = format_vram_clock(gpu.vram_freq_mhz, gpu.vram_max_freq_mhz);
                    format!(
                        "VRAM: {} / {} MiB{}",
                        gpu.vram_used_mb, gpu.vram_total_mb, vclk
                    )
                } else {
                    let uma_prefix = if gpu.vram_total_mb > 0 {
                        format!("UMA: {} / {} MiB | ", gpu.vram_used_mb, gpu.vram_total_mb)
                    } else {
                        String::new()
                    };
                    if m.ram_current_mts > 0 && m.ram_speed_mts > 0 {
                        format!(
                            "{uma_prefix}Unified Memory @ {} / {} MT/s",
                            m.ram_current_mts, m.ram_speed_mts
                        )
                    } else if m.ram_speed_mts > 0 {
                        format!("{uma_prefix}Unified Memory @ {} MT/s", m.ram_speed_mts)
                    } else {
                        format!("{uma_prefix}Unified Memory (Shared System RAM)")
                    }
                };

                let gpu_power_tag = if gpu.power_w > 0.0 {
                    format!(" | {:.1} W", gpu.power_w)
                } else {
                    String::new()
                };

                html.push_str(&format!(
                    "<b>GPU {idx} ({} [{}]):</b> {}% ({}/{} MHz) | {} °C{}<br/>{}<br/>",
                    gpu.name,
                    gpu.id,
                    gpu.load_pct,
                    gpu.freq_mhz,
                    gpu.max_freq_mhz,
                    gpu.temp_c,
                    gpu_power_tag,
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
// GUI Worker Window Process (Measured Pre-Map Sizing via egui::Context)
// ============================================================================

struct GuiWorkerState {
    snapshot: Mutex<SystemSnapshot>,
    interval_ms: AtomicU64,
    show_per_core: AtomicBool,
    egui_ctx: Mutex<Option<egui::Context>>,
    piped_mode: bool,
}

struct MonitorWindowApp {
    state: Arc<GuiWorkerState>,
    config: AppConfig,
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

        Self { state, config }
    }
}

fn render_monitor_contents(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    snap: &SystemSnapshot,
    config: &mut AppConfig,
    cur_interval: &mut u64,
    state: &GuiWorkerState,
    config_dirty: &mut bool,
    bar_w: f32,
) -> Option<[f32; 2]> {
    let mut fit_target = None;
    let bar_height = config.font_size + 6.0;

    let scale = config.font_size / 13.0;
    let mono_boost = if config.font_choice == FontChoice::Mono {
        1.15
    } else {
        1.0
    };
    let cols: usize = if snap.cpu_cores_mhz.is_empty() { 4 } else { 3 };
    let cell_w = if snap.cpu_cores_mhz.is_empty() {
        92.0 * scale * mono_boost
    } else {
        136.0 * scale * mono_boost
    };

    // --- TOP CONTROL BAR ---
    ui.horizontal(|ui| {
        ui.heading(RichText::new("KDE Monitor").strong());

        if snap.sys_power_w > 0.0 || snap.battery_pct.is_some() {
            let badge = match (snap.sys_power_w > 0.0, snap.battery_pct) {
                (true, Some(b)) if snap.sys_power_is_battery => {
                    format!("{:.1}W ({b}%)", snap.sys_power_w)
                }
                (true, Some(b)) => format!("{:.1}W (AC {b}%)", snap.sys_power_w),
                (true, None) => format!("{:.1}W", snap.sys_power_w),
                (false, Some(b)) => format!("AC {b}%"),
                (false, None) => String::new(),
            };
            if !badge.is_empty() {
                ui.label(
                    RichText::new(badge)
                        .color(Color32::from_rgb(255, 183, 77))
                        .strong(),
                );
            }
        }

        ui.separator();

        let prev_font = config.font_choice;
        egui::ComboBox::from_id_salt("font_family_combo")
            .selected_text(config.font_choice.label())
            .width(56.0 * (config.font_size / 13.0))
            .show_ui(ui, |ui| {
                for choice in FontChoice::ALL {
                    ui.selectable_value(&mut config.font_choice, choice, choice.label());
                }
            });
        if config.font_choice != prev_font {
            apply_font_style(ctx, config.font_choice, config.font_size);
            *config_dirty = true;
        }

        if ui
            .small_button("-")
            .on_hover_text("Decrease font size")
            .clicked()
        {
            config.font_size = (config.font_size - 1.0).max(9.0);
            apply_font_style(ctx, config.font_choice, config.font_size);
            *config_dirty = true;
        }
        ui.label(format!("{:.0}pt", config.font_size));
        if ui
            .small_button("+")
            .on_hover_text("Increase font size")
            .clicked()
        {
            config.font_size = (config.font_size + 1.0).min(24.0);
            apply_font_style(ctx, config.font_choice, config.font_size);
            *config_dirty = true;
        }

        ui.separator();

        egui::ComboBox::from_id_salt("poll_rate_combo")
            .selected_text(format!("{:.1}s", *cur_interval as f32 / 1000.0))
            .width(52.0 * (config.font_size / 13.0))
            .show_ui(ui, |ui| {
                for &ms in &POLL_INTERVALS_MS {
                    let label = format!("{:.1}s", ms as f32 / 1000.0);
                    if ui.selectable_value(cur_interval, ms, label).changed() {
                        state.interval_ms.store(*cur_interval, Ordering::Relaxed);
                        config.interval_ms = *cur_interval;
                        *config_dirty = true;
                        if state.piped_mode {
                            send_gui_control_msg(CTRL_SET_INTERVAL, *cur_interval);
                        }
                    }
                }
            });

        if ui.checkbox(&mut config.always_on_top, "Pin").changed() {
            let level = if config.always_on_top {
                WindowLevel::AlwaysOnTop
            } else {
                WindowLevel::Normal
            };
            ctx.send_viewport_cmd(ViewportCommand::WindowLevel(level));
            *config_dirty = true;
        }

        if ui
            .small_button("Fit")
            .on_hover_text("Compute fitted size for current layout")
            .clicked()
        {
            fit_target = Some(measure_viewport_size(config, snap));
        }

        if ui.button("Hide").clicked() {
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }
    });

    ui.separator();

    // --- CPU SECTION ---
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("CPU").strong());
            let cpu_meta = if snap.cpu_power_w > 0.0 {
                format!("{:.1} °C | {:.1} W", snap.cpu_temp_c, snap.cpu_power_w)
            } else {
                format!("{:.1} °C", snap.cpu_temp_c)
            };
            ui.label(RichText::new(cpu_meta).color(Color32::from_rgb(0, 188, 212)));
            if snap.cpu_avg_mhz > 0 {
                ui.label(
                    RichText::new(format!("• Avg: {}", format_mhz(snap.cpu_avg_mhz)))
                        .color(Color32::from_rgb(128, 222, 234)),
                );
            }
            if ui.checkbox(&mut config.show_per_core, "Per-Core").changed() {
                state
                    .show_per_core
                    .store(config.show_per_core, Ordering::Relaxed);
                if state.piped_mode {
                    send_gui_control_msg(CTRL_SET_PER_CORE, u64::from(config.show_per_core));
                }
                *config_dirty = true;
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
                .desired_width(bar_w)
                .desired_height(bar_height)
                .text(total_label)
                .fill(Color32::from_rgb(0, 150, 180)),
        );

        if config.show_per_core && !snap.cpu_cores_pct.is_empty() {
            ui.add_space(4.0);
            egui::Grid::new("cpu_cores_grid")
                .num_columns(cols)
                .spacing([6.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    for (idx, &pct) in snap.cpu_cores_pct.iter().enumerate() {
                        let cell_text = if let Some(&mhz) = snap.cpu_cores_mhz.get(idx) {
                            if mhz > 0 {
                                format!("C{idx}: {pct:.0}% • {:.2}G", mhz as f32 / 1000.0)
                            } else {
                                format!("C{idx}: {pct:.0}%")
                            }
                        } else {
                            format!("C{idx}: {pct:.0}%")
                        };

                        ui.add(
                            ProgressBar::new(pct / 100.0)
                                .desired_height(bar_height)
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
            let ram_badge = format_ram_meta(m.ram_current_mts, m.ram_speed_mts, m.ram_power_w);
            if !ram_badge.is_empty() {
                ui.label(RichText::new(ram_badge).color(Color32::from_rgb(186, 104, 200)));
            }
        });

        ui.add(
            ProgressBar::new(m.ram_pct / 100.0)
                .desired_width(bar_w)
                .desired_height(bar_height)
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
                    .desired_width(bar_w)
                    .desired_height(bar_height)
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
                ui.label(RichText::new(format!("ZRAM ({} [{}])", z.name, z.algorithm)).strong());
                ui.label(
                    RichText::new(format!("Ratio: {:.2}x", z.ratio))
                        .color(Color32::from_rgb(129, 199, 132)),
                );
            });
            ui.add(
                egui::Label::new(format!(
                    "Data: {} -> Compr: {} | RAM: {} / {}",
                    format_bytes(z.data_bytes),
                    format_bytes(z.compr_bytes),
                    format_bytes(z.total_ram_bytes),
                    format_bytes(z.disksize_bytes),
                ))
                .wrap_mode(egui::TextWrapMode::Extend),
            );
        }
    });

    // --- GPUS SECTION ---
    for (idx, gpu) in snap.gpus.iter().enumerate() {
        ui.add_space(4.0);
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("GPU {idx}: {} [{}]", gpu.name, gpu.id)).strong());
                let gpu_meta = if gpu.power_w > 0.0 {
                    format!("• {} °C | {:.1} W", gpu.temp_c, gpu.power_w)
                } else {
                    format!("• {} °C", gpu.temp_c)
                };
                ui.label(RichText::new(gpu_meta).color(Color32::from_rgb(102, 187, 106)));
            });

            ui.add(
                ProgressBar::new(gpu.load_pct as f32 / 100.0)
                    .desired_width(bar_w)
                    .desired_height(bar_height)
                    .text(format!(
                        "Clock Load: {}%  ({} / {} MHz)",
                        gpu.load_pct, gpu.freq_mhz, gpu.max_freq_mhz
                    ))
                    .fill(Color32::from_rgb(56, 142, 60)),
            );

            if !gpu.is_integrated && gpu.vram_total_mb > 0 {
                let vram_frac =
                    (gpu.vram_used_mb as f32 / gpu.vram_total_mb as f32).clamp(0.0, 1.0);
                let vclk_str = format_vram_clock(gpu.vram_freq_mhz, gpu.vram_max_freq_mhz);
                ui.add_space(2.0);
                ui.add(
                    ProgressBar::new(vram_frac)
                        .desired_width(bar_w)
                        .desired_height(bar_height)
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
                if gpu.vram_total_mb > 0 {
                    let uma_frac =
                        (gpu.vram_used_mb as f32 / gpu.vram_total_mb as f32).clamp(0.0, 1.0);
                    ui.add_space(2.0);
                    ui.add(
                        ProgressBar::new(uma_frac)
                            .desired_width(bar_w)
                            .desired_height(bar_height)
                            .text(format!(
                                "UMA Carveout: {} / {} MiB ({:.0}%)",
                                gpu.vram_used_mb,
                                gpu.vram_total_mb,
                                uma_frac * 100.0
                            ))
                            .fill(Color32::from_rgb(46, 110, 90)),
                    );
                }
                let unified_text = if snap.mem.ram_current_mts > 0 && snap.mem.ram_speed_mts > 0 {
                    format!(
                        "Unified Memory (Shared RAM @ {} / {} MT/s)",
                        snap.mem.ram_current_mts, snap.mem.ram_speed_mts
                    )
                } else if snap.mem.ram_speed_mts > 0 {
                    format!(
                        "Unified Memory (Shared RAM @ {} MT/s)",
                        snap.mem.ram_speed_mts
                    )
                } else {
                    "Unified Memory (Shared System RAM)".to_string()
                };
                ui.add(
                    egui::Label::new(
                        RichText::new(unified_text).color(Color32::from_rgb(180, 185, 195)),
                    )
                    .wrap_mode(egui::TextWrapMode::Extend),
                );
            }
        });
    }

    fit_target
}

/// Horizontal space `ui.group` consumes around its content (inner margin + stroke).
const GROUP_OVERHEAD: f32 = 14.0;

fn measure_viewport_size(config: &AppConfig, snap: &SystemSnapshot) -> [f32; 2] {
    let ctx = egui::Context::default();
    ctx.set_visuals(egui::Visuals::dark());
    setup_system_fonts(&ctx);
    apply_font_style(&ctx, config.font_choice, config.font_size);

    let sim_state = GuiWorkerState {
        snapshot: Mutex::new(snap.clone()),
        interval_ms: AtomicU64::new(config.interval_ms),
        show_per_core: AtomicBool::new(config.show_per_core),
        egui_ctx: Mutex::new(None),
        piped_mode: false,
    };

    let scale = config.font_size / 13.0;
    let mono_boost = if config.font_choice == FontChoice::Mono {
        1.15
    } else {
        1.0
    };
    let cols: usize = if snap.cpu_cores_mhz.is_empty() { 4 } else { 3 };
    let cell_w = if snap.cpu_cores_mhz.is_empty() {
        92.0 * scale * mono_boost
    } else {
        136.0 * scale * mono_boost
    };
    let grid_w = if config.show_per_core && !snap.cpu_cores_pct.is_empty() {
        (cell_w * cols as f32) + (6.0 * (cols.saturating_sub(1) as f32))
    } else {
        0.0
    };
    let target_bar_w = (440.0 * scale * mono_boost).max(grid_w);

    let raw_input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            Vec2::new(2400.0, 2400.0),
        )),
        ..Default::default()
    };

    // Fonts installed via `set_fonts` only take effect at the start of the next
    // pass, so run one empty pass first; otherwise the first measurement uses
    // default font metrics and the result is wrong.
    let _ = ctx.run(raw_input(), |_| {});

    // Several passes so state that egui carries over from the previous frame
    // (Grid column/row sizes, ComboBox, etc.) has settled before we read sizes.
    let mut measured = Vec2::ZERO;
    for _ in 0..3 {
        let _ = ctx.run(raw_input(), |ctx| {
            let _ = egui::CentralPanel::default().show(ctx, |ui| {
                let mut sim_cfg = config.clone();
                let mut sim_interval = config.interval_ms;
                let mut sim_dirty = false;

                // Bound the width so full-width widgets (separators, groups) don't
                // stretch to the 2400px sim screen. Wider items (top bar) overflow
                // and are still captured by `min_rect`.
                let mut child_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(egui::Rect::from_min_size(
                            ui.cursor().min,
                            Vec2::new(target_bar_w + GROUP_OVERHEAD, 2400.0),
                        ))
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                let _ = render_monitor_contents(
                    &mut child_ui,
                    ctx,
                    snap,
                    &mut sim_cfg,
                    &mut sim_interval,
                    &sim_state,
                    &mut sim_dirty,
                    target_bar_w,
                );
                measured = child_ui.min_rect().size();
            });
        });
    }

    let panel_frame = egui::Frame::central_panel(&ctx.style());
    let raw_w = measured.x + panel_frame.inner_margin.left + panel_frame.inner_margin.right + 2.0;
    let raw_h = measured.y + panel_frame.inner_margin.top + panel_frame.inner_margin.bottom;

    [raw_w.max(260.0).ceil(), raw_h.max(140.0).ceil()]
}

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

#[inline]
fn format_vram_clock(active_mhz: u32, max_mhz: u32) -> String {
    if active_mhz > 0 && max_mhz > 0 {
        format!(" @ {active_mhz} / {max_mhz} MHz")
    } else if active_mhz > 0 {
        format!(" @ {active_mhz} MHz")
    } else {
        String::new()
    }
}

#[inline]
fn format_ram_meta(cur_mts: u32, max_mts: u32, power_w: f32) -> String {
    let speed_str = if cur_mts > 0 && max_mts > 0 {
        format!("{cur_mts} / {max_mts} MT/s")
    } else if cur_mts > 0 {
        format!("{cur_mts} MT/s")
    } else if max_mts > 0 {
        format!("{max_mts} MT/s")
    } else {
        String::new()
    };

    match (!speed_str.is_empty(), power_w > 0.0) {
        (true, true) => format!("• {speed_str} | {power_w:.1} W"),
        (true, false) => format!("• {speed_str}"),
        (false, true) => format!("• {power_w:.1} W"),
        (false, false) => String::new(),
    }
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
        let mut fit_target = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            let bar_w = (ui.available_width() - GROUP_OVERHEAD).max(200.0);

            egui::ScrollArea::both()
                .auto_shrink([true, true])
                .show(ui, |ui| {
                    fit_target = render_monitor_contents(
                        ui,
                        ctx,
                        &snap,
                        &mut self.config,
                        &mut cur_interval,
                        &self.state,
                        &mut config_dirty,
                        bar_w,
                    );
                });
        });

        if let Some([w, h]) = fit_target {
            ctx.send_viewport_cmd(ViewportCommand::InnerSize(Vec2::new(w, h)));
        }

        if config_dirty {
            self.config.save();
        }
    }
}

fn run_gui_worker(piped_mode: bool, window_seq: u64) -> anyhow::Result<()> {
    let config = AppConfig::load();

    let initial_snapshot = if piped_mode {
        let mut stdin = io::stdin().lock();
        let Ok(snap) = decode_snapshot(&mut stdin) else {
            return Ok(());
        };
        snap
    } else {
        let mut collector = HardwareCollector::new()?;
        let mut snap = SystemSnapshot::default();
        collector.sample_into(&mut snap, config.show_per_core)?;
        snap
    };

    let initial_size = measure_viewport_size(&config, &initial_snapshot);
    let initial_digest = VisualDigest::from_snapshot(&initial_snapshot);
    let state = Arc::new(GuiWorkerState {
        snapshot: Mutex::new(initial_snapshot),
        interval_ms: AtomicU64::new(config.interval_ms),
        show_per_core: AtomicBool::new(config.show_per_core),
        egui_ctx: Mutex::new(None),
        piped_mode,
    });

    let state_bg = Arc::clone(&state);
    if piped_mode {
        thread::spawn(move || {
            let mut stdin = io::stdin().lock();
            let mut prev_digest = initial_digest;
            while let Ok(snap) = decode_snapshot(&mut stdin) {
                let digest = VisualDigest::from_snapshot(&snap);
                let per_core = state_bg.show_per_core.load(Ordering::Relaxed);
                if let Ok(mut guard) = state_bg.snapshot.lock() {
                    *guard = snap;
                }
                if digest != prev_digest || per_core {
                    prev_digest = digest;
                    if let Ok(guard) = state_bg.egui_ctx.lock() {
                        if let Some(ctx) = &*guard {
                            ctx.request_repaint();
                        }
                    }
                }
            }
        });
    } else {
        thread::spawn(move || {
            let Ok(mut collector) = HardwareCollector::new() else {
                return;
            };
            let mut local_snap = SystemSnapshot::default();
            let mut prev_digest = initial_digest;
            loop {
                let sleep_ms = state_bg.interval_ms.load(Ordering::Relaxed);
                thread::sleep(Duration::from_millis(sleep_ms));
                let per_core = state_bg.show_per_core.load(Ordering::Relaxed);
                if collector.sample_into(&mut local_snap, per_core).is_ok() {
                    let digest = VisualDigest::from_snapshot(&local_snap);
                    if let Ok(mut guard) = state_bg.snapshot.lock() {
                        *guard = local_snap.clone();
                    }
                    if digest != prev_digest || per_core {
                        prev_digest = digest;
                        if let Ok(guard) = state_bg.egui_ctx.lock() {
                            if let Some(ctx) = &*guard {
                                ctx.request_repaint();
                            }
                        }
                    }
                }
            }
        });
    }

    let app_id = format!("org.kde.rustmon.w{window_seq}");
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title("KDE Rust Hardware Monitor")
            .with_app_id(app_id)
            .with_resizable(true)
            .with_inner_size(initial_size)
            .with_min_inner_size([260.0, 140.0]),
        multisampling: 0,
        depth_buffer: 0,
        stencil_buffer: 0,
        ..Default::default()
    };

    let _ = eframe::run_native(
        "KDE Rust Hardware Monitor",
        native_options,
        Box::new(move |cc| Ok(Box::new(MonitorWindowApp::new(cc, state, config)))),
    );

    Ok(())
}

#[inline]
fn format_mhz(mhz: u32) -> String {
    if mhz >= 1000 {
        format!("{:.2} GHz", mhz as f32 / 1000.0)
    } else {
        format!("{mhz} MHz")
    }
}

#[inline]
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
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--gui-worker") {
        let piped = args.iter().any(|a| a == "--piped");
        let window_seq = args
            .iter()
            .find_map(|a| a.strip_prefix("--window-seq="))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        return run_gui_worker(piped, window_seq);
    }

    let saved_cfg = AppConfig::load();
    let start_hidden = args.iter().any(|a| a == "--tray" || a == "--hidden");
    let initial_per_core = !start_hidden && saved_cfg.show_per_core;

    let mut collector = HardwareCollector::new()?;
    let mut snapshot = SystemSnapshot::default();
    collector.sample_into(&mut snapshot, initial_per_core)?;
    let mut prev_digest = VisualDigest::from_snapshot(&snapshot);

    let shared = Arc::new(TraySharedState {
        interval_ms: AtomicU64::new(saved_cfg.interval_ms),
        window_seq: AtomicU64::new(1),
        collect_per_core_freqs: AtomicBool::new(initial_per_core),
        latest_snapshot: Mutex::new(snapshot.clone()),
        gui_client: Mutex::new(None),
        encode_buf: Mutex::new(Vec::with_capacity(1024)),
    });

    if !start_hidden {
        shared.open_window();
    }

    let tray = KdeResourceTray {
        snapshot: snapshot.clone(),
        shared: shared.clone(),
    };

    let service = TrayService::new(tray);
    let tray_handle = service.handle();
    service.spawn();

    loop {
        let sleep_ms = shared.interval_ms.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(sleep_ms));

        let per_core = shared.collect_per_core_freqs.load(Ordering::Relaxed);
        if collector.sample_into(&mut snapshot, per_core).is_ok() {
            if let Ok(mut guard) = shared.latest_snapshot.lock() {
                *guard = snapshot.clone();
            }

            shared.push_snapshot_to_gui(&snapshot);

            let digest = VisualDigest::from_snapshot(&snapshot);
            if digest != prev_digest {
                prev_digest = digest;
                let snap_for_tray = snapshot.clone();
                tray_handle.update(|tray: &mut KdeResourceTray| {
                    tray.snapshot = snap_for_tray;
                });
            }
        }
    }
}
