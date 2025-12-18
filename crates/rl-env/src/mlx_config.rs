use std::cell::Cell;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
struct MlxLimits {
    cache_limit_bytes: usize,
    memory_limit_bytes: Option<usize>,
    wired_limit_bytes: Option<usize>,
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn limits_from_env() -> MlxLimits {
    // Default to a conservative cache limit to avoid Metal resource exhaustion during long runs.
    // Override with AZUL_MLX_CACHE_LIMIT_BYTES (0 disables caching).
    let cache_limit_bytes = parse_env_usize("AZUL_MLX_CACHE_LIMIT_BYTES").unwrap_or(0);

    MlxLimits {
        cache_limit_bytes,
        memory_limit_bytes: parse_env_usize("AZUL_MLX_MEMORY_LIMIT_BYTES"),
        wired_limit_bytes: parse_env_usize("AZUL_MLX_WIRED_LIMIT_BYTES"),
    }
}

fn limits() -> &'static MlxLimits {
    static LIMITS: OnceLock<MlxLimits> = OnceLock::new();
    LIMITS.get_or_init(limits_from_env)
}

thread_local! {
    static CONFIGURED_FOR_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Apply MLX memory/cache limits for the current thread.
///
/// MLX settings may be thread-local depending on the backend; calling this once per thread keeps
/// behavior consistent across:
/// - the dedicated inference worker thread
/// - the training thread
/// - rayon self-play worker threads (feature encoding)
pub fn configure_mlx_for_current_thread() {
    CONFIGURED_FOR_THREAD.with(|configured| {
        if configured.get() {
            return;
        }
        configured.set(true);

        let limits = *limits();
        unsafe {
            let mut _prev = 0usize;
            let _ =
                mlx_sys::mlx_set_cache_limit(&mut _prev as *mut usize, limits.cache_limit_bytes);

            if let Some(limit) = limits.memory_limit_bytes {
                let mut _prev = 0usize;
                let _ = mlx_sys::mlx_set_memory_limit(&mut _prev as *mut usize, limit);
            }

            if let Some(limit) = limits.wired_limit_bytes {
                let mut _prev = 0usize;
                let _ = mlx_sys::mlx_set_wired_limit(&mut _prev as *mut usize, limit);
            }
        }
    });
}
