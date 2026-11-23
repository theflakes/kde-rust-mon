#![forbid(unsafe_code)]

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct MemoryMetrics {
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub ram_pct: f32,
    pub ram_speed_mts: u32,
    pub ram_current_mts: u32,
    pub ram_power_w: f32,
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
pub struct GpuMetrics {
    pub id: String,
    pub name: String,
    pub load_pct: u32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub vram_freq_mhz: u32,
    pub vram_max_freq_mhz: u32,
    pub temp_c: u32,
    pub power_w: f32,
    pub freq_mhz: u32,
    pub max_freq_mhz: u32,
}

#[derive(Clone, Debug, Default)]
pub struct SystemSnapshot {
    pub sys_power_w: f32,
    pub sys_power_is_battery: bool,
    pub battery_pct: Option<u8>,
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
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    steal: u64,
}

impl CpuTimes {
    fn total(self) -> u64 {
        self.user
            .saturating_add(self.nice)
            .saturating_add(self.system)
            .saturating_add(self.idle)
            .saturating_add(self.iowait)
            .saturating_add(self.irq)
            .saturating_add(self.softirq)
            .saturating_add(self.steal)
    }

    fn active(self) -> u64 {
        self.total()
            .saturating_sub(self.idle.saturating_add(self.iowait))
    }

    fn pct_since(self, prev: CpuTimes) -> f32 {
        let total_delta = self.total().saturating_sub(prev.total());
        if total_delta == 0 {
            return 0.0;
        }
        let active_delta = self.active().saturating_sub(prev.active());
        ((active_delta as f64 / total_delta as f64) * 100.0) as f32
    }
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
    fn read_watts(&mut self, buf: &mut [u8; 64]) -> f32 {
        match self {
            PowerSensor::DirectMicrowatts(f) => {
                let uw = read_u64_buf(f, buf).unwrap_or(0);
                uw as f32 / 1_000_000.0
            }
            PowerSensor::EnergyMicrojoules {
                file,
                prev_uj,
                prev_instant,
            } => {
                let now = Instant::now();
                let dt = now.duration_since(*prev_instant).as_secs_f64();
                let cur_uj = read_u64_buf(file, buf).unwrap_or(*prev_uj);
                let watts = if dt > 0.05 && cur_uj >= *prev_uj {
                    ((cur_uj - *prev_uj) as f64 / 1_000_000.0 / dt) as f32
                } else {
                    0.0
                };
                *prev_uj = cur_uj;
                *prev_instant = now;
                watts
            }
        }
    }
}

enum BatteryPowerRead {
    DirectPowerUw(File),
    VoltageCurrent { voltage_uv: File, current_ua: File },
}

struct BatterySensor {
    status_file: File,
    capacity_file: Option<File>,
    power_read: BatteryPowerRead,
}

impl BatterySensor {
    fn sample(&mut self, buf: &mut [u8; 64]) -> (Option<f32>, Option<u8>) {
        let cap = self
            .capacity_file
            .as_mut()
            .and_then(|f| read_u64_buf(f, buf))
            .map(|c| c.min(100) as u8);

        let is_discharging = read_str_buf(&mut self.status_file, buf)
            .map(|s| s.eq_ignore_ascii_case("Discharging"))
            .unwrap_or(false);

        if !is_discharging {
            return (None, cap);
        }

        let watts = match &mut self.power_read {
            BatteryPowerRead::DirectPowerUw(f) => {
                read_u64_buf(f, buf).unwrap_or(0) as f32 / 1_000_000.0
            }
            BatteryPowerRead::VoltageCurrent {
                voltage_uv,
                current_ua,
            } => {
                let v = read_u64_buf(voltage_uv, buf).unwrap_or(0) as f64 / 1_000_000.0;
                let a =
                    read_i64_buf(current_ua, buf).unwrap_or(0).unsigned_abs() as f64 / 1_000_000.0;
                (v * a) as f32
            }
        };

        if watts > 0.0 {
            (Some(watts), cap)
        } else {
            (None, cap)
        }
    }
}

struct ZramHandle {
    name: String,
    algorithm: String,
    disksize_file: File,
    mm_stat_file: File,
}

enum GpuSource {
    AmdDrm {
        id: String,
        name: String,
        busy_file: File,
        vram_used_file: Option<File>,
        vram_total_mb: u64,
        sclk_file: Option<File>,
        mclk_file: Option<File>,
        temp_file: Option<File>,
        power_sensor: Option<PowerSensor>,
    },
    IntelXe {
        id: String,
        name: String,
        cur_freq_file: File,
        max_freq_mhz: u32,
        temp_file: Option<File>,
        power_sensor: Option<PowerSensor>,
    },
    IntelI915 {
        id: String,
        name: String,
        cur_freq_file: File,
        max_freq_mhz: u32,
        temp_file: Option<File>,
        power_sensor: Option<PowerSensor>,
    },
    NvidiaNvml {
        index: u32,
        id: String,
        name: String,
    },
}

pub struct HardwareCollector {
    proc_stat: File,
    proc_meminfo: File,
    stat_buf: Vec<u8>,
    mem_buf: Vec<u8>,
    small_buf: [u8; 64],
    dpm_buf: [u8; 256],
    prev_total: CpuTimes,
    prev_cores: Vec<CpuTimes>,
    cpu_freq_files: Vec<Option<File>>,
    cpu_temp_file: Option<File>,
    cpu_power_sensor: Option<PowerSensor>,
    dram_power_sensor: Option<PowerSensor>,
    battery_sensor: Option<BatterySensor>,
    psys_power_sensor: Option<PowerSensor>,
    ram_speed_mts: u32,
    apu_mclk_file: Option<File>,
    zrams: Vec<ZramHandle>,
    nvml: Option<Nvml>,
    gpus: Vec<GpuSource>,
}

impl HardwareCollector {
    pub fn new() -> anyhow::Result<Self> {
        let mut proc_stat = File::open("/proc/stat")?;
        let proc_meminfo = File::open("/proc/meminfo")?;
        let mut stat_buf = Vec::with_capacity(8192);
        let mem_buf = Vec::with_capacity(4096);

        let mut prev_total = CpuTimes::default();
        let mut prev_cores = Vec::new();
        parse_proc_stat(
            &mut proc_stat,
            &mut stat_buf,
            &mut prev_total,
            &mut prev_cores,
        );

        let cpu_freq_files = Self::open_cpu_freq_files(prev_cores.len());
        let cpu_temp_file = Self::find_cpu_temp_sensor();
        let cpu_power_sensor = Self::find_cpu_power_sensor();
        let dram_power_sensor = Self::find_dram_power_sensor();
        let battery_sensor = Self::find_battery_sensor();
        let psys_power_sensor = Self::find_psys_sensor();
        let ram_speed_mts = detect_ram_speed_mts();
        let apu_mclk_file = Self::find_apu_mclk_file();
        let zrams = Self::discover_zram();
        let nvml = Nvml::init().ok();
        let gpus = Self::discover_gpus(nvml.as_ref());

        Ok(Self {
            proc_stat,
            proc_meminfo,
            stat_buf,
            mem_buf,
            small_buf: [0u8; 64],
            dpm_buf: [0u8; 256],
            prev_total,
            prev_cores,
            cpu_freq_files,
            cpu_temp_file,
            cpu_power_sensor,
            dram_power_sensor,
            battery_sensor,
            psys_power_sensor,
            ram_speed_mts,
            apu_mclk_file,
            zrams,
            nvml,
            gpus,
        })
    }

    pub fn sample_into(&mut self, snap: &mut SystemSnapshot) -> anyhow::Result<()> {
        // 1. CPU Utilization
        let mut cur_total = CpuTimes::default();
        let mut cur_cores = Vec::with_capacity(self.prev_cores.len());
        parse_proc_stat(
            &mut self.proc_stat,
            &mut self.stat_buf,
            &mut cur_total,
            &mut cur_cores,
        );

        snap.cpu_total_pct = cur_total.pct_since(self.prev_total);
        snap.cpu_cores_pct.clear();
        for (idx, &cur) in cur_cores.iter().enumerate() {
            let prev = self.prev_cores.get(idx).copied().unwrap_or(cur);
            snap.cpu_cores_pct.push(cur.pct_since(prev));
        }
        self.prev_total = cur_total;
        self.prev_cores = cur_cores;

        // 2. CPU Core Frequencies
        snap.cpu_cores_mhz.clear();
        let mut freq_sum = 0u64;
        let mut freq_count = 0u64;
        for opt_f in &mut self.cpu_freq_files {
            if let Some(f) = opt_f {
                let khz = read_u64_buf(f, &mut self.small_buf).unwrap_or(0);
                let mhz = (khz / 1000) as u32;
                snap.cpu_cores_mhz.push(mhz);
                if mhz > 0 {
                    freq_sum += mhz as u64;
                    freq_count += 1;
                }
            } else {
                snap.cpu_cores_mhz.push(0);
            }
        }
        snap.cpu_avg_mhz = if freq_count > 0 {
            (freq_sum / freq_count) as u32
        } else {
            0
        };

        // 3. CPU Temp & Power
        snap.cpu_temp_c = if let Some(ref mut f) = self.cpu_temp_file {
            read_u64_buf(f, &mut self.small_buf).unwrap_or(0) as f32 / 1000.0
        } else {
            0.0
        };

        snap.cpu_power_w = if let Some(ref mut sensor) = self.cpu_power_sensor {
            sensor.read_watts(&mut self.small_buf)
        } else {
            0.0
        };

        // 4. Total System Power (Battery discharge when unplugged, else RAPL psys)
        let psys_w = self
            .psys_power_sensor
            .as_mut()
            .map(|s| s.read_watts(&mut self.small_buf))
            .unwrap_or(0.0);

        if let Some(ref mut bat) = self.battery_sensor {
            let (bat_w, cap) = bat.sample(&mut self.small_buf);
            snap.battery_pct = cap;
            if let Some(w) = bat_w {
                snap.sys_power_w = w;
                snap.sys_power_is_battery = true;
            } else {
                snap.sys_power_w = psys_w;
                snap.sys_power_is_battery = false;
            }
        } else {
            snap.battery_pct = None;
            snap.sys_power_w = psys_w;
            snap.sys_power_is_battery = false;
        }

        // 5. Memory, Swap & DRAM Power
        parse_proc_meminfo(&mut self.proc_meminfo, &mut self.mem_buf, &mut snap.mem);
        snap.mem.ram_speed_mts = self.ram_speed_mts;
        snap.mem.ram_current_mts = if let Some(ref mut f) = self.apu_mclk_file {
            read_active_mclk_mts(f, &mut self.dpm_buf)
        } else {
            0
        };
        snap.mem.ram_power_w = if let Some(ref mut sensor) = self.dram_power_sensor {
            sensor.read_watts(&mut self.small_buf)
        } else {
            0.0
        };

        // 6. ZRAM Devices
        if snap.zrams.len() != self.zrams.len() {
            snap.zrams
                .resize_with(self.zrams.len(), ZramMetrics::default);
            for (dst, src) in snap.zrams.iter_mut().zip(self.zrams.iter()) {
                dst.name.clone_from(&src.name);
                dst.algorithm.clone_from(&src.algorithm);
            }
        }

        for (dst, src) in snap.zrams.iter_mut().zip(self.zrams.iter_mut()) {
            let disksize_bytes =
                read_u64_buf(&mut src.disksize_file, &mut self.small_buf).unwrap_or(0);
            let (data_bytes, compr_bytes, total_ram_bytes) =
                read_zram_mm_stat(&mut src.mm_stat_file, &mut self.dpm_buf);
            dst.disksize_bytes = disksize_bytes;
            dst.data_bytes = data_bytes;
            dst.compr_bytes = compr_bytes;
            dst.total_ram_bytes = total_ram_bytes;
            dst.ratio = if compr_bytes > 0 {
                data_bytes as f32 / compr_bytes as f32
            } else {
                1.0
            };
        }

        // 7. GPUs
        if snap.gpus.len() != self.gpus.len() {
            snap.gpus.resize_with(self.gpus.len(), GpuMetrics::default);
            for (dst, src) in snap.gpus.iter_mut().zip(self.gpus.iter()) {
                match src {
                    GpuSource::AmdDrm { id, name, .. }
                    | GpuSource::IntelXe { id, name, .. }
                    | GpuSource::IntelI915 { id, name, .. }
                    | GpuSource::NvidiaNvml { id, name, .. } => {
                        dst.id.clone_from(id);
                        dst.name.clone_from(name);
                    }
                }
            }
        }

        for (dst, src) in snap.gpus.iter_mut().zip(self.gpus.iter_mut()) {
            match src {
                GpuSource::AmdDrm {
                    busy_file,
                    vram_used_file,
                    vram_total_mb,
                    sclk_file,
                    mclk_file,
                    temp_file,
                    power_sensor,
                    ..
                } => {
                    dst.load_pct = read_u64_buf(busy_file, &mut self.small_buf).unwrap_or(0) as u32;
                    dst.vram_used_mb = if let Some(f) = vram_used_file {
                        read_u64_buf(f, &mut self.small_buf).unwrap_or(0) / (1024 * 1024)
                    } else {
                        0
                    };
                    dst.vram_total_mb = *vram_total_mb;

                    let (cur_sclk, max_sclk) = if let Some(f) = sclk_file {
                        read_pp_dpm_clocks(f, &mut self.dpm_buf)
                    } else {
                        (0, 0)
                    };
                    dst.freq_mhz = cur_sclk;
                    dst.max_freq_mhz = max_sclk;

                    let (cur_mclk, max_mclk) = if let Some(f) = mclk_file {
                        read_pp_dpm_clocks(f, &mut self.dpm_buf)
                    } else {
                        (0, 0)
                    };
                    dst.vram_freq_mhz = cur_mclk;
                    dst.vram_max_freq_mhz = max_mclk;

                    dst.temp_c = if let Some(f) = temp_file {
                        (read_u64_buf(f, &mut self.small_buf).unwrap_or(0) / 1000) as u32
                    } else {
                        0
                    };

                    dst.power_w = if let Some(sensor) = power_sensor {
                        sensor.read_watts(&mut self.small_buf)
                    } else {
                        0.0
                    };
                }
                GpuSource::IntelXe {
                    cur_freq_file,
                    max_freq_mhz,
                    temp_file,
                    power_sensor,
                    ..
                }
                | GpuSource::IntelI915 {
                    cur_freq_file,
                    max_freq_mhz,
                    temp_file,
                    power_sensor,
                    ..
                } => {
                    let cur_mhz =
                        read_u64_buf(cur_freq_file, &mut self.small_buf).unwrap_or(0) as u32;
                    dst.freq_mhz = cur_mhz;
                    dst.max_freq_mhz = *max_freq_mhz;
                    dst.load_pct = if *max_freq_mhz > 0 && cur_mhz > 100 {
                        ((cur_mhz as f32 / *max_freq_mhz as f32) * 100.0).round() as u32
                    } else {
                        0
                    };
                    dst.vram_used_mb = 0;
                    dst.vram_total_mb = 0;
                    dst.vram_freq_mhz = 0;
                    dst.vram_max_freq_mhz = 0;
                    dst.temp_c = if let Some(f) = temp_file {
                        (read_u64_buf(f, &mut self.small_buf).unwrap_or(0) / 1000) as u32
                    } else {
                        0
                    };
                    dst.power_w = if let Some(sensor) = power_sensor {
                        sensor.read_watts(&mut self.small_buf)
                    } else {
                        0.0
                    };
                }
                GpuSource::NvidiaNvml { index, .. } => {
                    if let Some(ref nvml) = self.nvml {
                        if let Ok(dev) = nvml.device_by_index(*index) {
                            dst.load_pct = dev.utilization_rates().map(|u| u.gpu).unwrap_or(0);
                            if let Ok(mem) = dev.memory_info() {
                                dst.vram_used_mb = mem.used / (1024 * 1024);
                                dst.vram_total_mb = mem.total / (1024 * 1024);
                            }
                            dst.freq_mhz = dev.clock_info(Clock::Graphics).unwrap_or(0);
                            dst.max_freq_mhz = dev.max_clock_info(Clock::Graphics).unwrap_or(0);
                            dst.vram_freq_mhz = dev.clock_info(Clock::Memory).unwrap_or(0);
                            dst.vram_max_freq_mhz = dev.max_clock_info(Clock::Memory).unwrap_or(0);
                            dst.temp_c = dev.temperature(TemperatureSensor::Gpu).unwrap_or(0);
                            dst.power_w = dev
                                .power_usage()
                                .map(|mw| mw as f32 / 1000.0)
                                .unwrap_or(0.0);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn open_cpu_freq_files(num_cores: usize) -> Vec<Option<File>> {
        (0..num_cores)
            .map(|i| {
                let p = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq");
                File::open(p).ok()
            })
            .collect()
    }

    fn find_cpu_temp_sensor() -> Option<File> {
        let entries = fs::read_dir("/sys/class/hwmon").ok()?;
        let mut fallback = None;
        for entry in entries.flatten() {
            let p = entry.path();
            let Ok(name) = fs::read_to_string(p.join("name")) else {
                continue;
            };
            let n = name.trim();
            if n == "k10temp" || n == "zenpower" || n == "coretemp" {
                if let Ok(f) = File::open(p.join("temp1_input")) {
                    return Some(f);
                }
            } else if (n == "acpitz" || n == "thinkpad" || n == "dell_smm") && fallback.is_none() {
                fallback = File::open(p.join("temp1_input")).ok();
            }
        }
        fallback
    }

    fn find_cpu_power_sensor() -> Option<PowerSensor> {
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

        let mut f = File::open("/sys/class/powercap/intel-rapl:0/energy_uj").ok()?;
        let prev_uj = read_u64_fast(&mut f).unwrap_or(0);
        Some(PowerSensor::EnergyMicrojoules {
            file: f,
            prev_uj,
            prev_instant: Instant::now(),
        })
    }

    fn find_dram_power_sensor() -> Option<PowerSensor> {
        let rapl_root = Path::new("/sys/class/powercap/intel-rapl:0");
        let entries = fs::read_dir(rapl_root).ok()?;
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(name) = fs::read_to_string(p.join("name")) {
                if name.trim() == "dram" {
                    if let Ok(mut f) = File::open(p.join("energy_uj")) {
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
        None
    }

    fn find_battery_sensor() -> Option<BatterySensor> {
        let entries = fs::read_dir("/sys/class/power_supply").ok()?;
        let mut bats: Vec<PathBuf> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                if name.to_string_lossy().starts_with("BAT") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        bats.sort();

        for bat in bats {
            let Ok(status_file) = File::open(bat.join("status")) else {
                continue;
            };
            let capacity_file = File::open(bat.join("capacity")).ok();
            if let Ok(power_f) = File::open(bat.join("power_now")) {
                return Some(BatterySensor {
                    status_file,
                    capacity_file,
                    power_read: BatteryPowerRead::DirectPowerUw(power_f),
                });
            }
            if let (Ok(voltage_uv), Ok(current_ua)) = (
                File::open(bat.join("voltage_now")),
                File::open(bat.join("current_now")),
            ) {
                return Some(BatterySensor {
                    status_file,
                    capacity_file,
                    power_read: BatteryPowerRead::VoltageCurrent {
                        voltage_uv,
                        current_ua,
                    },
                });
            }
        }
        None
    }

    fn find_psys_sensor() -> Option<PowerSensor> {
        let entries = fs::read_dir("/sys/class/powercap").ok()?;
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(name) = fs::read_to_string(p.join("name")) {
                if name.trim() == "psys" {
                    if let Ok(mut f) = File::open(p.join("energy_uj")) {
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
        None
    }

    fn find_apu_mclk_file() -> Option<File> {
        let entries = fs::read_dir("/sys/class/drm").ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if !s.starts_with("card") || s.contains('-') {
                continue;
            }
            let dev = entry.path().join("device");
            let vendor = fs::read_to_string(dev.join("vendor")).unwrap_or_default();
            let class = fs::read_to_string(dev.join("class")).unwrap_or_default();
            if vendor.trim() == "0x1002" && class.trim().starts_with("0x0380") {
                if let Ok(f) = File::open(dev.join("pp_dpm_mclk")) {
                    return Some(f);
                }
            }
        }
        None
    }

    fn discover_zram() -> Vec<ZramHandle> {
        let mut list = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/block") else {
            return list;
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter_map(|e| {
                let s = e.file_name().to_string_lossy().into_owned();
                if s.starts_with("zram") { Some(s) } else { None }
            })
            .collect();
        names.sort();

        for name in names {
            let base = PathBuf::from("/sys/block").join(&name);
            let Ok(mut disksize_file) = File::open(base.join("disksize")) else {
                continue;
            };
            if read_u64_fast(&mut disksize_file).unwrap_or(0) == 0 {
                continue;
            }
            let Ok(mm_stat_file) = File::open(base.join("mm_stat")) else {
                continue;
            };
            let algorithm = fs::read_to_string(base.join("comp_algorithm"))
                .ok()
                .and_then(|s| {
                    s.split_whitespace()
                        .find(|tok| tok.starts_with('[') && tok.ends_with(']'))
                        .map(|tok| tok.trim_matches(&['[', ']'][..]).to_string())
                })
                .unwrap_or_else(|| "zstd".to_string());

            list.push(ZramHandle {
                name,
                algorithm,
                disksize_file,
                mm_stat_file,
            });
        }
        list
    }

    fn discover_gpus(nvml: Option<&Nvml>) -> Vec<GpuSource> {
        let mut gpus = Vec::new();

        if let Some(nvml) = nvml {
            if let Ok(count) = nvml.device_count() {
                for idx in 0..count {
                    if let Ok(dev) = nvml.device_by_index(idx) {
                        let name = dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {idx}"));
                        gpus.push(GpuSource::NvidiaNvml {
                            index: idx,
                            id: format!("nvml:{idx}"),
                            name,
                        });
                    }
                }
            }
        }

        let Ok(entries) = fs::read_dir("/sys/class/drm") else {
            return gpus;
        };
        let mut cards: Vec<String> = entries
            .flatten()
            .filter_map(|e| {
                let s = e.file_name().to_string_lossy().into_owned();
                if s.starts_with("card") && !s.contains('-') {
                    Some(s)
                } else {
                    None
                }
            })
            .collect();
        cards.sort();

        for card in cards {
            let dev_dir = PathBuf::from("/sys/class/drm").join(&card).join("device");
            let vendor = fs::read_to_string(dev_dir.join("vendor"))
                .unwrap_or_default()
                .trim()
                .to_string();
            let dev_id = fs::read_to_string(dev_dir.join("device"))
                .unwrap_or_default()
                .trim()
                .trim_start_matches("0x")
                .to_string();

            if vendor == "0x10de" && !gpus.is_empty() {
                continue;
            }

            if vendor == "0x1002" {
                if let Ok(busy_file) = File::open(dev_dir.join("gpu_busy_percent")) {
                    let vram_used_file = File::open(dev_dir.join("mem_info_vram_used")).ok();
                    let vram_total_mb = File::open(dev_dir.join("mem_info_vram_total"))
                        .ok()
                        .and_then(|mut f| read_u64_fast(&mut f))
                        .unwrap_or(0)
                        / (1024 * 1024);
                    let sclk_file = File::open(dev_dir.join("pp_dpm_sclk")).ok();
                    let mclk_file = File::open(dev_dir.join("pp_dpm_mclk")).ok();
                    let (temp_file, power_sensor) = Self::find_hwmon_sensors(&dev_dir, &vendor);
                    let name = if dev_id.is_empty() {
                        format!("AMD Radeon ({card})")
                    } else {
                        format!("AMD Radeon ({dev_id})")
                    };
                    gpus.push(GpuSource::AmdDrm {
                        id: card,
                        name,
                        busy_file,
                        vram_used_file,
                        vram_total_mb,
                        sclk_file,
                        mclk_file,
                        temp_file,
                        power_sensor,
                    });
                }
            } else if vendor == "0x8086" {
                let xe_cur = dev_dir.join("tile0/gt0/freq0/act_freq");
                let xe_max = dev_dir.join("tile0/gt0/freq0/max_freq");
                if let Ok(cur_freq_file) = File::open(&xe_cur) {
                    let max_freq_mhz = File::open(&xe_max)
                        .ok()
                        .and_then(|mut f| read_u64_fast(&mut f))
                        .unwrap_or(2500) as u32;
                    let (temp_file, power_sensor) = Self::find_hwmon_sensors(&dev_dir, &vendor);
                    let name = if dev_id.is_empty() {
                        format!("Intel Xe ({card})")
                    } else {
                        format!("Intel Xe ({dev_id})")
                    };
                    gpus.push(GpuSource::IntelXe {
                        id: card,
                        name,
                        cur_freq_file,
                        max_freq_mhz,
                        temp_file,
                        power_sensor,
                    });
                } else {
                    let i915_cur = PathBuf::from("/sys/class/drm")
                        .join(&card)
                        .join("gt_act_freq_mhz");
                    let i915_max = PathBuf::from("/sys/class/drm")
                        .join(&card)
                        .join("gt_max_freq_mhz");
                    if let Ok(cur_freq_file) = File::open(&i915_cur) {
                        let max_freq_mhz = File::open(&i915_max)
                            .ok()
                            .and_then(|mut f| read_u64_fast(&mut f))
                            .unwrap_or(1500) as u32;
                        let (temp_file, power_sensor) = Self::find_hwmon_sensors(&dev_dir, &vendor);
                        let name = if dev_id.is_empty() {
                            format!("Intel Graphics ({card})")
                        } else {
                            format!("Intel Graphics ({dev_id})")
                        };
                        gpus.push(GpuSource::IntelI915 {
                            id: card,
                            name,
                            cur_freq_file,
                            max_freq_mhz,
                            temp_file,
                            power_sensor,
                        });
                    }
                }
            }
        }

        gpus
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
            let power = Self::find_intel_igpu_rapl_sensor();
            return (temp, power);
        }

        (None, None)
    }

    fn find_intel_igpu_rapl_sensor() -> Option<PowerSensor> {
        let rapl_root = Path::new("/sys/class/powercap/intel-rapl:0");
        let entries = fs::read_dir(rapl_root).ok()?;
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(name) = fs::read_to_string(p.join("name")) {
                let n = name.trim();
                if n == "uncore" || n == "gpu" || n == "gt" {
                    if let Ok(mut f) = File::open(p.join("energy_uj")) {
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
        None
    }
}

// ============================================================================
// Stack-Buffered Sysfs / Procfs Parsers
// ============================================================================

#[inline]
fn read_u64_fast(file: &mut File) -> Option<u64> {
    let mut buf = [0u8; 64];
    read_u64_buf(file, &mut buf)
}

#[inline]
fn read_u64_buf(file: &mut File, buf: &mut [u8; 64]) -> Option<u64> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let n = file.read(buf).ok()?;
    let s = std::str::from_utf8(&buf[..n]).ok()?.trim();
    s.parse::<u64>().ok()
}

#[inline]
fn read_i64_buf(file: &mut File, buf: &mut [u8; 64]) -> Option<i64> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let n = file.read(buf).ok()?;
    let s = std::str::from_utf8(&buf[..n]).ok()?.trim();
    s.parse::<i64>().ok()
}

#[inline]
fn read_str_buf<'a>(file: &mut File, buf: &'a mut [u8; 64]) -> Option<&'a str> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let n = file.read(buf).ok()?;
    std::str::from_utf8(&buf[..n]).ok().map(str::trim)
}

fn parse_proc_stat(
    file: &mut File,
    buf: &mut Vec<u8>,
    total_out: &mut CpuTimes,
    cores_out: &mut Vec<CpuTimes>,
) {
    buf.clear();
    if file.seek(SeekFrom::Start(0)).is_err() || file.read_to_end(buf).is_err() {
        return;
    }
    let text = match std::str::from_utf8(buf) {
        Ok(t) => t,
        Err(_) => return,
    };

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("cpu") {
            let mut it = rest.split_whitespace();
            let first = it.next().unwrap_or("");
            if rest.starts_with(' ') {
                *total_out = parse_cpu_fields(first, it);
            } else if first.chars().all(|c| c.is_ascii_digit()) && !first.is_empty() {
                let mut fields = rest.split_whitespace();
                let _cpu_idx = fields.next();
                let first_val = fields.next().unwrap_or("0");
                cores_out.push(parse_cpu_fields(first_val, fields));
            }
        } else if !cores_out.is_empty() {
            break;
        }
    }
}

#[inline]
fn parse_cpu_fields<'a, I: Iterator<Item = &'a str>>(first: &str, mut it: I) -> CpuTimes {
    CpuTimes {
        user: first.parse().unwrap_or(0),
        nice: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
        system: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
        idle: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
        iowait: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
        irq: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
        softirq: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
        steal: it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
    }
}

fn parse_proc_meminfo(file: &mut File, buf: &mut Vec<u8>, mem: &mut MemoryMetrics) {
    buf.clear();
    if file.seek(SeekFrom::Start(0)).is_err() || file.read_to_end(buf).is_err() {
        return;
    }
    let text = match std::str::from_utf8(buf) {
        Ok(t) => t,
        Err(_) => return,
    };

    let mut mem_total_kb = 0u64;
    let mut mem_avail_kb = 0u64;
    let mut swap_total_kb = 0u64;
    let mut swap_free_kb = 0u64;
    let mut found = 0u8;

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            mem_total_kb = parse_kb_val(rest);
            found |= 0b0001;
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            mem_avail_kb = parse_kb_val(rest);
            found |= 0b0010;
        } else if let Some(rest) = line.strip_prefix("SwapTotal:") {
            swap_total_kb = parse_kb_val(rest);
            found |= 0b0100;
        } else if let Some(rest) = line.strip_prefix("SwapFree:") {
            swap_free_kb = parse_kb_val(rest);
            found |= 0b1000;
        }
        if found == 0b1111 {
            break;
        }
    }

    let mem_used_kb = mem_total_kb.saturating_sub(mem_avail_kb);
    mem.ram_total_mb = mem_total_kb / 1024;
    mem.ram_used_mb = mem_used_kb / 1024;
    mem.ram_pct = if mem_total_kb > 0 {
        (mem_used_kb as f64 / mem_total_kb as f64 * 100.0) as f32
    } else {
        0.0
    };

    let swap_used_kb = swap_total_kb.saturating_sub(swap_free_kb);
    mem.swap_total_mb = swap_total_kb / 1024;
    mem.swap_used_mb = swap_used_kb / 1024;
    mem.swap_pct = if swap_total_kb > 0 {
        (swap_used_kb as f64 / swap_total_kb as f64 * 100.0) as f32
    } else {
        0.0
    };
}

#[inline]
fn parse_kb_val(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

fn read_zram_mm_stat(file: &mut File, buf: &mut [u8; 256]) -> (u64, u64, u64) {
    if file.seek(SeekFrom::Start(0)).is_err() {
        return (0, 0, 0);
    }
    let Ok(n) = file.read(buf) else {
        return (0, 0, 0);
    };
    let s = std::str::from_utf8(&buf[..n]).unwrap_or("");
    let mut it = s.split_whitespace();
    let orig_data = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let compr_data = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let mem_used_total = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    (orig_data, compr_data, mem_used_total)
}

fn read_pp_dpm_clocks(file: &mut File, buf: &mut [u8; 256]) -> (u32, u32) {
    if file.seek(SeekFrom::Start(0)).is_err() {
        return (0, 0);
    }
    let Ok(n) = file.read(buf) else {
        return (0, 0);
    };
    let s = std::str::from_utf8(&buf[..n]).unwrap_or("");
    let mut active = 0u32;
    let mut max_mhz = 0u32;
    for line in s.lines() {
        if let Some(mhz_tok) = line.split_whitespace().nth(1) {
            let mhz: u32 = mhz_tok
                .trim_end_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .unwrap_or(0);
            if mhz > max_mhz {
                max_mhz = mhz;
            }
            if line.contains('*') {
                active = mhz;
            }
        }
    }
    (active, max_mhz)
}

fn read_active_mclk_mts(file: &mut File, buf: &mut [u8; 256]) -> u32 {
    let (active_mhz, _) = read_pp_dpm_clocks(file, buf);
    active_mhz * 2
}

fn detect_ram_speed_mts() -> u32 {
    if let Ok(dmi) = fs::read("/sys/firmware/dmi/tables/DMI") {
        let mut i = 0usize;
        let mut best_speed = 0u32;
        while i + 4 <= dmi.len() {
            let rec_type = dmi[i];
            let rec_len = dmi[i + 1] as usize;
            if rec_len < 4 || i + rec_len > dmi.len() {
                break;
            }
            if rec_type == 17 && rec_len >= 0x17 {
                let size_field = u16::from_le_bytes([dmi[i + 0x0C], dmi[i + 0x0D]]);
                if size_field != 0 && size_field != 0xFFFF {
                    let speed = u16::from_le_bytes([dmi[i + 0x15], dmi[i + 0x16]]) as u32;
                    let conf_speed = if rec_len >= 0x22 {
                        u16::from_le_bytes([dmi[i + 0x20], dmi[i + 0x21]]) as u32
                    } else {
                        0
                    };
                    let active = if conf_speed > 0 && conf_speed < 65535 {
                        conf_speed
                    } else if speed > 0 && speed < 65535 {
                        speed
                    } else {
                        0
                    };
                    if active > best_speed {
                        best_speed = active;
                    }
                }
            }
            let mut next = i + rec_len;
            while next + 1 < dmi.len() && !(dmi[next] == 0 && dmi[next + 1] == 0) {
                next += 1;
            }
            i = next + 2;
        }
        if best_speed > 0 {
            return best_speed;
        }
    }

    if let Ok(udev_dmi) = fs::read_to_string("/run/udev/data/+dmi:id") {
        let mut best_speed = 0u32;
        for line in udev_dmi.lines() {
            if let Some(rest) = line.strip_prefix("E:MEMORY_DEVICE_") {
                if let Some((_, val)) = rest.split_once("_CONFIGURED_SPEED_MTS=") {
                    if let Ok(mts) = val.trim().parse::<u32>() {
                        if mts > best_speed && mts < 65535 {
                            best_speed = mts;
                        }
                    }
                } else if let Some((_, val)) = rest.split_once("_SPEED_MTS=") {
                    if let Ok(mts) = val.trim().parse::<u32>() {
                        if mts > best_speed && mts < 65535 {
                            best_speed = mts;
                        }
                    }
                }
            }
        }
        if best_speed > 0 {
            return best_speed;
        }
    }

    0
}
