use anyhow::Result;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct GpuMetrics {
    pub id: String,
    pub name: String,
    pub load_pct: u32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub vram_freq_mhz: u32,
    pub temp_c: u32,
    pub power_w: f32,
    pub freq_mhz: u32,
    pub max_freq_mhz: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryMetrics {
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub ram_pct: f32,
    pub ram_speed_mts: u32,
    pub swap_used_mb: u64,
    pub swap_total_mb: u64,
    pub swap_pct: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ZramMetrics {
    pub name: String,
    pub algorithm: String,
    pub disksize_bytes: u64,
    pub data_bytes: u64,
    pub compr_bytes: u64,
    pub total_ram_bytes: u64,
    pub ratio: f32,
}

#[derive(Clone, Debug, Default)]
pub struct SystemSnapshot {
    pub cpu_total_pct: f32,
    pub cpu_cores_pct: Vec<f32>,
    pub cpu_avg_mhz: u32,
    pub cpu_cores_mhz: Vec<u32>,
    pub cpu_temp_c: f32,
    pub cpu_power_w: f32,
    pub mem: MemoryMetrics,
    pub zrams: Vec<ZramMetrics>,
    pub gpus: Vec<GpuMetrics>,
}

#[derive(Clone, Copy, Default)]
struct CpuTimes {
    active: u64,
    total: u64,
}

enum ClockSensor {
    DirectFreqFile {
        act_file: File,
        max_mhz: u32,
    },
    AmdDpmSclk {
        sclk_file: File,
        fallback_max_mhz: u32,
    },
}

enum VramClockSensor {
    AmdDpmMclk(File),
}

enum PowerSensor {
    DirectMicrowatts(File),
    EnergyMicrojoules {
        file: File,
        prev_uj: u64,
        prev_instant: Instant,
    },
}

impl PowerSensor {
    fn sample_watts(&mut self, now: Instant) -> f32 {
        match self {
            PowerSensor::DirectMicrowatts(file) => read_u64_fast(file)
                .map(|uw| uw as f32 / 1_000_000.0)
                .unwrap_or(0.0),
            PowerSensor::EnergyMicrojoules {
                file,
                prev_uj,
                prev_instant,
            } => {
                let dt = now.duration_since(*prev_instant).as_secs_f32();
                *prev_instant = now;
                if let Ok(cur_uj) = read_u64_fast(file) {
                    let delta_uj = cur_uj.wrapping_sub(*prev_uj);
                    *prev_uj = cur_uj;
                    if dt > 0.05 {
                        (delta_uj as f32 / 1_000_000.0) / dt
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
        }
    }
}

struct SysfsGpuHandle {
    id: String,
    name: String,
    clock_sensor: ClockSensor,
    vram_clock_sensor: Option<VramClockSensor>,
    vram_used_file: Option<File>,
    vram_total_mb: u64,
    temp_file: Option<File>,
    power_sensor: Option<PowerSensor>,
}

struct ZramHandle {
    name: String,
    algorithm: String,
    disksize_bytes: u64,
    mm_stat_file: File,
}

pub struct HardwareCollector {
    proc_stat: File,
    proc_meminfo: File,
    proc_cpuinfo: Option<File>,
    cpu_freq_files: Vec<File>,
    stat_buf: Vec<u8>,
    prev_cpu_total: CpuTimes,
    prev_cpu_cores: Vec<CpuTimes>,
    cpu_temp_file: Option<File>,
    cpu_power_sensor: Option<PowerSensor>,
    ram_speed_mts: u32,
    zram_handles: Vec<ZramHandle>,
    nvml: Option<Nvml>,
    sysfs_gpus: Vec<SysfsGpuHandle>,
}

impl HardwareCollector {
    pub fn new() -> Result<Self> {
        let proc_stat = File::open("/proc/stat")?;
        let proc_meminfo = File::open("/proc/meminfo")?;
        let cpu_freq_files = Self::discover_cpu_freq_files();
        let proc_cpuinfo = if cpu_freq_files.is_empty() {
            File::open("/proc/cpuinfo").ok()
        } else {
            None
        };
        let cpu_temp_file = Self::find_cpu_temp_sensor();
        let cpu_power_sensor = Self::find_cpu_power_sensor();
        let ram_speed_mts = Self::detect_ram_speed_mts();
        let zram_handles = Self::discover_zram_devices();
        let nvml = Nvml::init().ok();
        let sysfs_gpus = Self::discover_sysfs_gpus();

        let mut collector = Self {
            proc_stat,
            proc_meminfo,
            proc_cpuinfo,
            cpu_freq_files,
            stat_buf: Vec::with_capacity(8192),
            prev_cpu_total: CpuTimes::default(),
            prev_cpu_cores: Vec::new(),
            cpu_temp_file,
            cpu_power_sensor,
            ram_speed_mts,
            zram_handles,
            nvml,
            sysfs_gpus,
        };

        let _ = collector.sample_cpu();
        Ok(collector)
    }

    pub fn sample(&mut self) -> Result<SystemSnapshot> {
        let now = Instant::now();
        let (cpu_total_pct, cpu_cores_pct) = self.sample_cpu()?;
        let (cpu_avg_mhz, cpu_cores_mhz) = self.sample_cpu_freqs();
        let mem = self.sample_memory()?;
        let zrams = self.sample_zram();
        let cpu_temp_c = self
            .cpu_temp_file
            .as_mut()
            .and_then(|f| read_u64_fast(f).ok())
            .map(|millideg| millideg as f32 / 1000.0)
            .unwrap_or(0.0);

        let cpu_power_w = self
            .cpu_power_sensor
            .as_mut()
            .map(|s| s.sample_watts(now))
            .unwrap_or(0.0);

        let mut gpus = Vec::new();
        self.sample_nvml_gpus(&mut gpus);
        self.sample_sysfs_gpus(&mut gpus, now);

        Ok(SystemSnapshot {
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

    fn discover_cpu_freq_files() -> Vec<File> {
        let mut entries_vec: Vec<(usize, PathBuf)> = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
            return Vec::new();
        };

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(num_str) = name_str.strip_prefix("cpu") {
                if !num_str.is_empty() && num_str.bytes().all(|b| b.is_ascii_digit()) {
                    if let Ok(idx) = num_str.parse::<usize>() {
                        let freq_path = entry.path().join("cpufreq/scaling_cur_freq");
                        if freq_path.exists() {
                            entries_vec.push((idx, freq_path));
                        }
                    }
                }
            }
        }

        entries_vec.sort_by_key(|(idx, _)| *idx);

        entries_vec
            .into_iter()
            .filter_map(|(_, path)| File::open(path).ok())
            .collect()
    }

    fn sample_cpu_freqs(&mut self) -> (u32, Vec<u32>) {
        if !self.cpu_freq_files.is_empty() {
            let mut per_core = Vec::with_capacity(self.cpu_freq_files.len());
            let mut sum_mhz = 0u64;
            let mut active_count = 0u64;

            for file in &mut self.cpu_freq_files {
                let mhz = read_u64_fast(file)
                    .map(|khz| (khz / 1000) as u32)
                    .unwrap_or(0);
                if mhz > 0 {
                    sum_mhz += mhz as u64;
                    active_count += 1;
                }
                per_core.push(mhz);
            }

            let avg_mhz = if active_count > 0 {
                (sum_mhz / active_count) as u32
            } else {
                0
            };

            return (avg_mhz, per_core);
        }

        if let Some(file) = &mut self.proc_cpuinfo {
            if file.seek(SeekFrom::Start(0)).is_ok() {
                self.stat_buf.clear();
                if file.read_to_end(&mut self.stat_buf).is_ok() {
                    if let Ok(content) = std::str::from_utf8(&self.stat_buf) {
                        let mut per_core = Vec::new();
                        let mut sum_mhz = 0u64;
                        for line in content.lines() {
                            if let Some(rest) = line.strip_prefix("cpu MHz") {
                                if let Some((_, val_str)) = rest.split_once(':') {
                                    let mhz =
                                        val_str.trim().parse::<f32>().unwrap_or(0.0).round() as u32;
                                    sum_mhz += mhz as u64;
                                    per_core.push(mhz);
                                }
                            }
                        }
                        let avg = if !per_core.is_empty() {
                            (sum_mhz / per_core.len() as u64) as u32
                        } else {
                            0
                        };
                        return (avg, per_core);
                    }
                }
            }
        }

        (0, Vec::new())
    }

    fn sample_cpu(&mut self) -> Result<(f32, Vec<f32>)> {
        self.proc_stat.seek(SeekFrom::Start(0))?;
        self.stat_buf.clear();
        self.proc_stat.read_to_end(&mut self.stat_buf)?;

        let content = std::str::from_utf8(&self.stat_buf)?;
        let mut total_pct = 0.0;
        let mut cores_pct = Vec::with_capacity(self.prev_cpu_cores.len());
        let mut core_idx = 0;

        for line in content.lines() {
            if !line.starts_with("cpu") {
                break;
            }
            let mut parts = line.split_ascii_whitespace();
            let label = parts.next().unwrap_or("");

            let mut vals = [0u64; 10];
            for (i, part) in parts.take(10).enumerate() {
                vals[i] = part.parse().unwrap_or(0);
            }

            let active = vals[0] + vals[1] + vals[2] + vals[5] + vals[6] + vals[7];
            let idle = vals[3] + vals[4];
            let total = active + idle;
            let current = CpuTimes { active, total };

            if label == "cpu" {
                total_pct = calc_cpu_delta(self.prev_cpu_total, current);
                self.prev_cpu_total = current;
            } else {
                if core_idx >= self.prev_cpu_cores.len() {
                    self.prev_cpu_cores.push(current);
                    cores_pct.push(0.0);
                } else {
                    let pct = calc_cpu_delta(self.prev_cpu_cores[core_idx], current);
                    self.prev_cpu_cores[core_idx] = current;
                    cores_pct.push(pct);
                }
                core_idx += 1;
            }
        }

        Ok((total_pct, cores_pct))
    }

    fn sample_memory(&mut self) -> Result<MemoryMetrics> {
        self.proc_meminfo.seek(SeekFrom::Start(0))?;
        self.stat_buf.clear();
        self.proc_meminfo.read_to_end(&mut self.stat_buf)?;

        let content = std::str::from_utf8(&self.stat_buf)?;
        let mut mem_total_kb = 0u64;
        let mut mem_avail_kb = 0u64;
        let mut swap_total_kb = 0u64;
        let mut swap_free_kb = 0u64;
        let mut found = 0u8;

        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                mem_total_kb = parse_kb(rest);
                found |= 1;
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                mem_avail_kb = parse_kb(rest);
                found |= 2;
            } else if let Some(rest) = line.strip_prefix("SwapTotal:") {
                swap_total_kb = parse_kb(rest);
                found |= 4;
            } else if let Some(rest) = line.strip_prefix("SwapFree:") {
                swap_free_kb = parse_kb(rest);
                found |= 8;
            }
            if found == 0b1111 {
                break;
            }
        }

        let ram_used_kb = mem_total_kb.saturating_sub(mem_avail_kb);
        let swap_used_kb = swap_total_kb.saturating_sub(swap_free_kb);

        let ram_pct = if mem_total_kb > 0 {
            (ram_used_kb as f32 / mem_total_kb as f32) * 100.0
        } else {
            0.0
        };

        let swap_pct = if swap_total_kb > 0 {
            (swap_used_kb as f32 / swap_total_kb as f32) * 100.0
        } else {
            0.0
        };

        Ok(MemoryMetrics {
            ram_used_mb: ram_used_kb / 1024,
            ram_total_mb: mem_total_kb / 1024,
            ram_pct,
            ram_speed_mts: self.ram_speed_mts,
            swap_used_mb: swap_used_kb / 1024,
            swap_total_mb: swap_total_kb / 1024,
            swap_pct,
        })
    }

    fn detect_ram_speed_mts() -> u32 {
        let Ok(entries) = fs::read_dir("/sys/firmware/dmi/entries") else {
            return 0;
        };

        let mut max_speed = 0u32;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with("17-") {
                continue;
            }

            let Ok(raw) = fs::read(entry.path().join("raw")) else {
                continue;
            };

            if raw.len() < 0x17 {
                continue;
            }
            let size = u16::from_le_bytes([raw[0x0C], raw[0x0D]]);
            if size == 0 {
                continue;
            }

            let nominal_speed = u16::from_le_bytes([raw[0x15], raw[0x16]]) as u32;
            let configured_speed = if raw.len() >= 0x22 {
                u16::from_le_bytes([raw[0x20], raw[0x21]]) as u32
            } else {
                0
            };

            let speed = if configured_speed > 0 && configured_speed != 0xFFFF {
                configured_speed
            } else if nominal_speed != 0xFFFF {
                nominal_speed
            } else {
                0
            };

            max_speed = max_speed.max(speed);
        }

        max_speed
    }

    fn discover_zram_devices() -> Vec<ZramHandle> {
        let mut zrams = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/block") else {
            return zrams;
        };

        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("zram") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        paths.sort();

        for path in paths {
            let disksize_bytes = fs::read_to_string(path.join("disksize"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);

            if disksize_bytes == 0 {
                continue;
            }

            let Ok(mm_stat_file) = File::open(path.join("mm_stat")) else {
                continue;
            };

            let raw_algo = fs::read_to_string(path.join("comp_algorithm")).unwrap_or_default();
            let algorithm = parse_bracketed_algo(&raw_algo);

            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();

            zrams.push(ZramHandle {
                name,
                algorithm,
                disksize_bytes,
                mm_stat_file,
            });
        }

        zrams
    }

    fn sample_zram(&mut self) -> Vec<ZramMetrics> {
        let mut out = Vec::with_capacity(self.zram_handles.len());
        let mut buf = [0u8; 256];

        for z in &mut self.zram_handles {
            if z.mm_stat_file.seek(SeekFrom::Start(0)).is_err() {
                continue;
            }
            let Ok(n) = z.mm_stat_file.read(&mut buf) else {
                continue;
            };
            let Ok(s) = std::str::from_utf8(&buf[..n]) else {
                continue;
            };

            let mut parts = s.split_ascii_whitespace();
            let data_bytes = parts
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let compr_bytes = parts
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let total_ram_bytes = parts
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);

            let ratio = if compr_bytes > 0 {
                data_bytes as f32 / compr_bytes as f32
            } else {
                0.0
            };

            out.push(ZramMetrics {
                name: z.name.clone(),
                algorithm: z.algorithm.clone(),
                disksize_bytes: z.disksize_bytes,
                data_bytes,
                compr_bytes,
                total_ram_bytes,
                ratio,
            });
        }

        out
    }

    fn sample_nvml_gpus(&self, out: &mut Vec<GpuMetrics>) {
        let Some(nvml) = &self.nvml else { return };
        let count = nvml.device_count().unwrap_or(0);

        for i in 0..count {
            if let Ok(dev) = nvml.device_by_index(i) {
                let name = dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}"));
                let (vram_used_mb, vram_total_mb) = dev
                    .memory_info()
                    .map(|m| (m.used / 1_048_576, m.total / 1_048_576))
                    .unwrap_or((0, 0));
                let temp_c = dev.temperature(TemperatureSensor::Gpu).unwrap_or(0);
                let power_w = dev
                    .power_usage()
                    .map(|mw| mw as f32 / 1000.0)
                    .unwrap_or(0.0);

                let freq_mhz = dev.clock_info(Clock::Graphics).unwrap_or(0);
                let max_freq_mhz = dev.max_clock_info(Clock::Graphics).unwrap_or(0);
                let vram_freq_mhz = dev.clock_info(Clock::Memory).unwrap_or(0);

                let load_pct = if max_freq_mhz > 0 {
                    ((freq_mhz as f32 / max_freq_mhz as f32) * 100.0)
                        .round()
                        .clamp(0.0, 100.0) as u32
                } else {
                    dev.utilization_rates().map(|u| u.gpu).unwrap_or(0)
                };

                out.push(GpuMetrics {
                    id: format!("nvml:{i}"),
                    name,
                    load_pct,
                    vram_used_mb,
                    vram_total_mb,
                    vram_freq_mhz,
                    temp_c,
                    power_w,
                    freq_mhz,
                    max_freq_mhz,
                });
            }
        }
    }

    fn sample_sysfs_gpus(&mut self, out: &mut Vec<GpuMetrics>, now: Instant) {
        for gpu in &mut self.sysfs_gpus {
            let (freq_mhz, max_freq_mhz) = match &mut gpu.clock_sensor {
                ClockSensor::DirectFreqFile { act_file, max_mhz } => {
                    let act = read_u64_fast(act_file).unwrap_or(0) as u32;
                    (act, *max_mhz)
                }
                ClockSensor::AmdDpmSclk {
                    sclk_file,
                    fallback_max_mhz,
                } => {
                    let (act, max) = parse_amd_dpm_clock(sclk_file);
                    (act, if max > 0 { max } else { *fallback_max_mhz })
                }
            };

            let load_pct = if max_freq_mhz > 0 {
                ((freq_mhz as f32 / max_freq_mhz as f32) * 100.0)
                    .round()
                    .clamp(0.0, 100.0) as u32
            } else {
                0
            };

            let vram_freq_mhz = match &mut gpu.vram_clock_sensor {
                Some(VramClockSensor::AmdDpmMclk(mclk_file)) => parse_amd_dpm_clock(mclk_file).0,
                None => 0,
            };

            let vram_used_mb = gpu
                .vram_used_file
                .as_mut()
                .and_then(|f| read_u64_fast(f).ok())
                .map(|b| b / 1_048_576)
                .unwrap_or(0);

            let temp_c = gpu
                .temp_file
                .as_mut()
                .and_then(|f| read_u64_fast(f).ok())
                .map(|m| (m / 1000) as u32)
                .unwrap_or(0);

            let power_w = gpu
                .power_sensor
                .as_mut()
                .map(|s| s.sample_watts(now))
                .unwrap_or(0.0);

            out.push(GpuMetrics {
                id: gpu.id.clone(),
                name: gpu.name.clone(),
                load_pct,
                vram_used_mb,
                vram_total_mb: gpu.vram_total_mb,
                vram_freq_mhz,
                temp_c,
                power_w,
                freq_mhz,
                max_freq_mhz,
            });
        }
    }

    fn discover_sysfs_gpus() -> Vec<SysfsGpuHandle> {
        let mut gpus = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/class/drm") else {
            return gpus;
        };

        let mut cards: Vec<PathBuf> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("card") && !name.contains('-') {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        cards.sort();

        for card_path in cards {
            let card_name = card_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let dev_dir = card_path.join("device");

            let vendor = fs::read_to_string(dev_dir.join("vendor"))
                .unwrap_or_default()
                .trim()
                .to_lowercase();

            if vendor == "0x10de" {
                continue;
            }

            let Some(clock_sensor) = Self::find_clock_sensor(&card_path, &dev_dir) else {
                continue;
            };

            let vram_clock_sensor = File::open(dev_dir.join("pp_dpm_mclk"))
                .ok()
                .map(VramClockSensor::AmdDpmMclk);

            let vram_total_mb = fs::read_to_string(dev_dir.join("mem_info_vram_total"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|b| b / 1_048_576)
                .unwrap_or(0);

            let (temp_file, power_sensor) = Self::find_hwmon_sensors(&dev_dir, &vendor);
            let friendly_name = Self::resolve_gpu_name(&dev_dir, &vendor, &card_name);

            gpus.push(SysfsGpuHandle {
                id: card_name,
                name: friendly_name,
                clock_sensor,
                vram_clock_sensor,
                vram_used_file: File::open(dev_dir.join("mem_info_vram_used")).ok(),
                vram_total_mb,
                temp_file,
                power_sensor,
            });
        }
        gpus
    }

    fn find_clock_sensor(card_path: &Path, dev_dir: &Path) -> Option<ClockSensor> {
        let xe_freq_dir = dev_dir.join("tile0/gt0/freq0");
        if let Ok(act_file) = File::open(xe_freq_dir.join("act_freq")) {
            let max_mhz = ["max_freq", "rp0_freq", "rpe_freq"]
                .iter()
                .find_map(|name| {
                    fs::read_to_string(xe_freq_dir.join(name))
                        .ok()
                        .and_then(|s| s.trim().parse::<u32>().ok())
                        .filter(|&v| v > 0)
                })
                .unwrap_or(0);
            return Some(ClockSensor::DirectFreqFile { act_file, max_mhz });
        }

        if let Ok(act_file) = File::open(card_path.join("gt/gt0/rps_act_freq_mhz"))
            .or_else(|_| File::open(card_path.join("gt_act_freq_mhz")))
        {
            let max_mhz = [
                card_path.join("gt/gt0/rps_max_freq_mhz"),
                card_path.join("gt/gt0/rps_RP0_freq_mhz"),
                card_path.join("gt_max_freq_mhz"),
                card_path.join("gt_RP0_freq_mhz"),
            ]
            .iter()
            .find_map(|p| {
                fs::read_to_string(p)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .filter(|&v| v > 0)
            })
            .unwrap_or(0);
            return Some(ClockSensor::DirectFreqFile { act_file, max_mhz });
        }

        if let Ok(sclk_file) = File::open(dev_dir.join("pp_dpm_sclk")) {
            return Some(ClockSensor::AmdDpmSclk {
                sclk_file,
                fallback_max_mhz: 0,
            });
        }

        None
    }

    fn resolve_gpu_name(dev_dir: &Path, vendor: &str, card_name: &str) -> String {
        let device_id = fs::read_to_string(dev_dir.join("device"))
            .unwrap_or_default()
            .trim()
            .trim_start_matches("0x")
            .to_lowercase();

        if vendor == "0x8086" {
            return format!("Intel Xe ({device_id})");
        } else if vendor == "0x1002" {
            return format!("AMD Radeon ({card_name})");
        }
        format!("DRM {}", card_name.to_uppercase())
    }

    fn find_hwmon_sensors(dev_dir: &Path, vendor: &str) -> (Option<File>, Option<PowerSensor>) {
        let hwmon_base = dev_dir.join("hwmon");
        if let Ok(entries) = fs::read_dir(hwmon_base) {
            for entry in entries.flatten() {
                let p = entry.path();
                let temp = (1..=4).find_map(|i| File::open(p.join(format!("temp{i}_input"))).ok());

                let power = if let Ok(f) = File::open(p.join("power1_average"))
                    .or_else(|_| File::open(p.join("power1_input")))
                {
                    Some(PowerSensor::DirectMicrowatts(f))
                } else if let Ok(mut f) = File::open(p.join("energy1_input"))
                    .or_else(|_| File::open(p.join("energy2_input")))
                {
                    let prev_uj = read_u64_fast(&mut f).unwrap_or(0);
                    Some(PowerSensor::EnergyMicrojoules {
                        file: f,
                        prev_uj,
                        prev_instant: Instant::now(),
                    })
                } else {
                    None
                };

                if temp.is_some() || power.is_some() {
                    return (temp, power);
                }
            }
        }

        if vendor == "0x8086" {
            let temp = Self::find_cpu_temp_sensor();
            let power = Self::find_intel_rapl_sensor(true);
            return (temp, power);
        }

        (None, None)
    }

    /// Discovers CPU package/core wattage via Intel RAPL (`powercap`) or AMD `hwmon` (`k10temp`/`zenpower`).
    fn find_cpu_power_sensor() -> Option<PowerSensor> {
        // 1. Check AMD/Intel hwmon direct power/energy sensors first
        if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Ok(name) = fs::read_to_string(path.join("name")) {
                    let n = name.trim();
                    if n == "k10temp" || n == "zenpower" || n == "coretemp" {
                        if let Ok(f) = File::open(path.join("power1_average"))
                            .or_else(|_| File::open(path.join("power1_input")))
                        {
                            return Some(PowerSensor::DirectMicrowatts(f));
                        }
                        if let Ok(mut f) = File::open(path.join("energy1_input")) {
                            let prev_uj = read_u64_fast(&mut f).unwrap_or(0);
                            return Some(PowerSensor::EnergyMicrojoules {
                                file: f,
                                prev_uj,
                                prev_instant: Instant::now(),
                            });
                        }
                    }
                }
            }
        }

        // 2. Intel RAPL Package Power (`package-0` at `/sys/class/powercap/intel-rapl:0/energy_uj`)
        Self::find_intel_rapl_sensor(false)
    }

    /// If `prefer_uncore` is true, picks the `uncore` (iGPU) sub-domain if exposed, falling back to `package-0`.
    /// If `prefer_uncore` is false, reads `package-0` directly for total CPU package wattage.
    fn find_intel_rapl_sensor(prefer_uncore: bool) -> Option<PowerSensor> {
        let rapl_root = Path::new("/sys/class/powercap/intel-rapl:0");
        if !rapl_root.exists() {
            return None;
        }

        let mut chosen_energy_path = None;
        if prefer_uncore {
            if let Ok(entries) = fs::read_dir(rapl_root) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Ok(name) = fs::read_to_string(p.join("name")) {
                        if name.trim() == "uncore" {
                            chosen_energy_path = Some(p.join("energy_uj"));
                            break;
                        }
                    }
                }
            }
        }

        let energy_path = chosen_energy_path.unwrap_or_else(|| rapl_root.join("energy_uj"));
        let mut f = File::open(energy_path).ok()?;
        let prev_uj = read_u64_fast(&mut f).unwrap_or(0);
        Some(PowerSensor::EnergyMicrojoules {
            file: f,
            prev_uj,
            prev_instant: Instant::now(),
        })
    }

    fn find_cpu_temp_sensor() -> Option<File> {
        let entries = fs::read_dir("/sys/class/hwmon").ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(name) = fs::read_to_string(path.join("name")) {
                let n = name.trim();
                if n == "coretemp" || n == "k10temp" || n == "zenpower" || n == "acpitz" {
                    if let Ok(file) = File::open(path.join("temp1_input")) {
                        return Some(file);
                    }
                }
            }
        }
        None
    }
}

fn parse_amd_dpm_clock(file: &mut File) -> (u32, u32) {
    if file.seek(SeekFrom::Start(0)).is_err() {
        return (0, 0);
    }
    let mut buf = [0u8; 512];
    let Ok(n) = file.read(&mut buf) else {
        return (0, 0);
    };
    let Ok(content) = std::str::from_utf8(&buf[..n]) else {
        return (0, 0);
    };

    let mut active_mhz = 0u32;
    let mut max_mhz = 0u32;

    for line in content.lines() {
        if let Some((_, rest)) = line.split_once(':') {
            let trimmed = rest.trim();
            let is_active = trimmed.ends_with('*');
            let mhz = trimmed
                .split(|c: char| !c.is_ascii_digit())
                .find(|s| !s.is_empty())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);

            max_mhz = max_mhz.max(mhz);
            if is_active {
                active_mhz = mhz;
            }
        }
    }

    (active_mhz, max_mhz)
}

#[inline]
fn parse_bracketed_algo(s: &str) -> String {
    if let Some(start) = s.find('[') {
        if let Some(end) = s[start + 1..].find(']') {
            return s[start + 1..start + 1 + end].to_string();
        }
    }
    s.trim().to_string()
}

#[inline]
fn calc_cpu_delta(prev: CpuTimes, curr: CpuTimes) -> f32 {
    let total_delta = curr.total.saturating_sub(prev.total);
    if total_delta == 0 {
        return 0.0;
    }
    let active_delta = curr.active.saturating_sub(prev.active);
    (active_delta as f32 / total_delta as f32) * 100.0
}

#[inline]
fn read_u64_fast(file: &mut File) -> Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut buf = [0u8; 32];
    let n = file.read(&mut buf)?;
    let s = std::str::from_utf8(&buf[..n])?.trim();
    Ok(s.parse()?)
}

#[inline]
fn parse_kb(s: &str) -> u64 {
    s.split_ascii_whitespace()
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}
