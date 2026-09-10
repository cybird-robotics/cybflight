//! Python bridge: a batch of [`RecoveryEnv`]s stepped in parallel.
//!
//! `VecEnv.step(actions)` steps every environment on a rayon pool and
//! auto-resets the ones that finished, returning the terminal observation
//! in `infos` the way SB3's `VecEnv` contract expects.

use cybflight_core::mpc::cost_adapt::{COST_MAP_VERSION, NZ, OBS_DIM};
use cybflight_sim::rl_env::{EnvConfig, RecoveryEnv, RewardWeights};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass]
struct VecEnv {
    envs: Vec<RecoveryEnv>,
}

fn load_vehicle(path: &str) -> PyResult<(cybflight_core::params::FirmwareConfig, vehicle_yaml::SimYaml)> {
    let yaml = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("vehicle");
    let v = vehicle_yaml::load(name, &yaml)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{path}: {e:?}")))?;
    // Plant-only physics come from the frozen sim baseline when the
    // vehicle YAML carries no `sim:` section (flight YAMLs never do).
    let sim = cybflight_sim::scenario::default_sim_params();
    Ok((v.params, sim))
}

#[pymethods]
impl VecEnv {
    /// `n` environments on `vehicle_yaml`, seeds `seed..seed+n`.
    #[new]
    #[pyo3(signature = (n, vehicle_yaml, seed=0, blind=false, disturb=1.0, mass_jitter=0.2,
                        max_episode_s=4.0, w_geom=10.0, w_time=1.0, w_rate=0.2, w_dz=0.1, w_crash=20.0, w_du_rate=25.0, gyro_sigma=0.03, rate_weight_nominal=5.0, sampler_max_lag=0.1, model_drag=false, body_drag=0.0, domain_rand=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        vehicle_yaml: &str,
        seed: u64,
        blind: bool,
        disturb: f32,
        mass_jitter: f32,
        max_episode_s: f32,
        w_geom: f32,
        w_time: f32,
        w_rate: f32,
        w_dz: f32,
        w_crash: f32,
        w_du_rate: f32,
        gyro_sigma: f32,
        rate_weight_nominal: f32,
        sampler_max_lag: f32,
        model_drag: bool,
        body_drag: f32,
        domain_rand: f32,
    ) -> PyResult<Self> {
        let (vp, sim) = load_vehicle(vehicle_yaml)?;
        let mut cfg = EnvConfig::new(vp, sim);
        cfg.blind = blind;
        cfg.disturb = disturb;
        cfg.mass_jitter = mass_jitter;
        cfg.max_episode_s = max_episode_s;
        cfg.reward = RewardWeights { geom: w_geom, time: w_time, rate: w_rate, dz: w_dz, crash: w_crash, du_rate: w_du_rate };
        cfg.gyro_sigma = gyro_sigma;
        cfg.rate_weight_nominal = rate_weight_nominal;
        cfg.sampler_max_lag_s = sampler_max_lag;
        cfg.model_drag = model_drag;
        cfg.body_drag = [body_drag, body_drag, 2.0 * body_drag];
        cfg.domain_rand = domain_rand;
        let envs = (0..n)
            .map(|i| RecoveryEnv::new(cfg.clone(), seed + i as u64))
            .collect();
        Ok(Self { envs })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.envs.len()
    }
    #[getter]
    fn obs_dim(&self) -> usize {
        OBS_DIM
    }
    #[getter]
    fn act_dim(&self) -> usize {
        NZ
    }

    /// Reset every environment; returns `(n, OBS_DIM)`.
    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let obs: Vec<[f32; OBS_DIM]> = py.detach(|| self.envs.par_iter_mut().map(|e| e.reset()).collect());
        flat2(py, &obs)
    }

    /// Step every environment with `actions (n, NZ)`. Returns
    /// `(obs, rewards, dones, infos)`; a finished environment is reset and
    /// its terminal observation is stored in `infos["terminal_obs"]`.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyDict>,
    )> {
        let a = actions.as_array();
        if a.shape() != [self.envs.len(), NZ] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "actions must be ({}, {NZ}), got {:?}",
                self.envs.len(),
                a.shape()
            )));
        }
        let acts: Vec<[f32; NZ]> = (0..self.envs.len())
            .map(|i| core::array::from_fn(|j| a[[i, j]]))
            .collect();
        struct Out {
            obs: [f32; OBS_DIM],
            terminal_obs: [f32; OBS_DIM],
            r: f32,
            done: bool,
            e_geom: f32,
            e_time: f32,
            du_rate2: f32,
            crashed: bool,
            truncated: bool,
            diverged: bool,
            kind: u8,
            speed: f32,
            peak_thrust: f32,
            peak_rate: f32,
        }
        let outs: Vec<Out> = py.detach(|| {
            self.envs
                .par_iter_mut()
                .zip(acts.par_iter())
                .map(|(e, z)| {
                    let (obs, r, done, info) = e.step(z);
                    let ep = e.episode;
                    let (obs, terminal_obs) = if done { (e.reset(), obs) } else { (obs, [0.0; OBS_DIM]) };
                    Out {
                        obs,
                        terminal_obs,
                        r,
                        done,
                        e_geom: info.e_geom,
                        e_time: info.e_time,
                        du_rate2: info.du_rate2,
                        crashed: info.crashed,
                        truncated: info.truncated,
                        diverged: info.diverged,
                        kind: ep.kind,
                        speed: ep.speed,
                        peak_thrust: ep.peak_thrust_frac,
                        peak_rate: ep.peak_rate_frac,
                    }
                })
                .collect()
        });
        let obs: Vec<[f32; OBS_DIM]> = outs.iter().map(|o| o.obs).collect();
        let tobs: Vec<[f32; OBS_DIM]> = outs.iter().map(|o| o.terminal_obs).collect();
        let infos = PyDict::new(py);
        infos.set_item("terminal_obs", flat2(py, &tobs))?;
        infos.set_item("e_geom", outs.iter().map(|o| o.e_geom).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("e_time", outs.iter().map(|o| o.e_time).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("du_rate2", outs.iter().map(|o| o.du_rate2).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("crashed", outs.iter().map(|o| o.crashed).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("truncated", outs.iter().map(|o| o.truncated).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("diverged", outs.iter().map(|o| o.diverged).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("kind", outs.iter().map(|o| o.kind).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("speed", outs.iter().map(|o| o.speed).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("peak_thrust", outs.iter().map(|o| o.peak_thrust).collect::<Vec<_>>().into_pyarray(py))?;
        infos.set_item("peak_rate", outs.iter().map(|o| o.peak_rate).collect::<Vec<_>>().into_pyarray(py))?;
        Ok((
            flat2(py, &obs),
            outs.iter().map(|o| o.r).collect::<Vec<_>>().into_pyarray(py),
            outs.iter().map(|o| o.done).collect::<Vec<_>>().into_pyarray(py),
            infos,
        ))
    }
}

fn flat2<'py, const D: usize>(py: Python<'py>, rows: &[[f32; D]]) -> Bound<'py, PyArray2<f32>> {
    PyArray2::from_vec2(py, &rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
        .expect("rows are uniform")
}

#[pymodule]
fn cybflight_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add("OBS_DIM", OBS_DIM)?;
    m.add("NZ", NZ)?;
    // The exporter stamps this into the checkpoint header; the firmware
    // bake refuses a checkpoint whose map revision is not the one the
    // firmware was compiled against.
    m.add("COST_MAP_VERSION", COST_MAP_VERSION)?;
    Ok(())
}
