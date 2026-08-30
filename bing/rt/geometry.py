"""Fixed per-pixel viewing/illumination geometry for the robust.rt backend.

See ``docs/design/rob_rt_design.md`` §3.2 for the design this implements,
and ``claude_prompts/rob_rt.md`` (Q&A/Design items 3, 11, 14) for why
geometry is fixed, non-fit data rather than a new MCMC parameter: it is a
known/measured scene quantity, not something the sampler should search
over.
"""
from dataclasses import dataclass

from robust.rt import types as robust_types


@dataclass(frozen=True)
class ObsGeometry:
    """Fixed per-pixel viewing/illumination geometry, in degrees. Never fit.

    ``theta_s`` is required and never defaulted (claude_prompts/rob_rt.md,
    Q&A/Coding item 4): a robust-backend fit with no geometry supplied must
    raise at fit setup rather than silently assume an angle.
    ``theta_v``/``dphi`` default to nadir viewing (0, 0) when only a solar
    zenith is known.

    Attributes:
        theta_s (float): Solar zenith angle (degrees); 0 = sun overhead.
            Required -- no default.
        theta_v (float): Sensor zenith angle (degrees); 0 = nadir view.
            Defaults to nadir.
        dphi (float): Sensor-sun relative azimuth (degrees). Defaults to 0.
        wind (float, optional): Wind speed (m/s), optional surface-roughness
            input. ``None`` until a reference dataset varies it.
    """
    theta_s: float
    theta_v: float = 0.0
    dphi: float = 0.0
    wind: float | None = None

    def to_robust(self, Ed=None):
        """Map onto ``robust.rt.types.Geometry`` (same units, degrees).

        Args:
            Ed (tuple, optional): A ``(wave_Ed, Ed)`` pass-through pair for
                robust's downwelling-irradiance override -- wired in M4.
                Not stored on ``ObsGeometry``; the dataclass stays pure
                per-pixel metadata, so callers supply it explicitly at the
                point of use.

        Returns:
            robust.rt.types.Geometry: the equivalent robust geometry.
        """
        return robust_types.Geometry(
            theta_s=self.theta_s, theta_v=self.theta_v, dphi=self.dphi,
            wind=self.wind, Ed=Ed)
