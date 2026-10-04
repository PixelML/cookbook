// Real-time solar ephemeris.
// Source: US Naval Observatory / Meeus, Astronomical Algorithms (ch. 25).
// All angles in degrees. Input time is UTC (epoch milliseconds or Date).

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

/** Julian Date from a UTC epoch (ms). */
export function julianDate(ms) {
  return ms / 86400000 + 2440587.5;
}

/** Julian centuries since J2000.0. */
export function julianCenturies(ms) {
  return (julianDate(ms) - 2451545.0) / 36525;
}

/** Sun geocentric apparent ecliptic longitude (deg) and distance (AU). */
export function sunEcliptic(ms) {
  const T = julianCenturies(ms);

  // Mean longitude, mean anomaly (deg).
  const L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T;
  const M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
  const Mr = M * DEG;

  // Equation of centre.
  const C =
    (1.914602 - 0.004817 * T - 0.000014 * T * T) * Math.sin(Mr) +
    (0.019993 - 0.000310 * T) * Math.sin(2 * Mr) +
    0.001010 * Math.sin(3 * Mr);

  const lambda = L0 + C; // apparent ecliptic longitude
  const trueAnomaly = M + C;
  const R =
    1.000001 * (1 - 0.016713 * Math.cos(Mr) - 0.0001395 * Math.cos(2 * Mr));

  // Obliquity of the ecliptic (deg).
  const eps = 23.439291 - 0.0130042 * T - 0.00000016 * T * T + 0.0000005 * T * T * T;

  return { T, lambda: norm360(lambda), eps, R, trueAnomaly };
}

/** Right ascension (deg), declination (deg), subsolar point (deg). */
export function solarPosition(ms) {
  const { T, lambda, eps, R } = sunEcliptic(ms);
  const lr = lambda * DEG, er = eps * DEG;

  const ra = Math.atan2(Math.cos(er) * Math.sin(lr), Math.cos(lr)); // radians
  const dec = Math.asin(Math.sin(er) * Math.sin(lr)); // radians

  const gmst = greenwichMeanSiderealTime(ms); // degrees
  const gast = gmst + equationOfEquinoxes(T); // degrees

  // Subsolar longitude: the meridian where the sun transits (local hour angle = 0).
  const lon = norm180(ra * RAD - gast);

  return {
    declination: dec * RAD,
    rightAscension: norm360(ra * RAD),
    gmst,
    gast,
    subsolar: { lat: dec * RAD, lon },
    distanceAU: R,
    // Sun direction in the Earth-fixed frame used by latLonToUnit (same handedness as the UV map).
    dir: latLonToUnit(dec * RAD, lon),
  };
}

/** Greenwich Mean Sidereal Time in degrees. */
export function greenwichMeanSiderealTime(ms) {
  const d = julianDate(ms) - 2451545.0;
  const theta = 18.69737456 + 24.065709824419 * d; // hours
  return norm360(theta * 15);
}

/** Equation of the equinoxes (deg), for apparent sidereal time. */
function equationOfEquinoxes(T) {
  const Omega = 125.04452 - 1934.136261 * T;
  return 0.00468 * Math.sin(Omega * DEG) + 0.000009 * Math.sin(2 * Omega * DEG);
}

function norm360(x) {
  return ((x % 360) + 360) % 360;
}

function norm180(x) {
  const v = norm360(x);
  return v > 180 ? v - 360 : v;
}

/** lat/lon (deg) -> unit vector in the same Earth-fixed frame as solarPosition.dir. */
export function latLonToUnit(lat, lon) {
  const p = lat * DEG, l = lon * DEG;
  return { x: Math.cos(p) * Math.cos(l), y: Math.sin(p), z: -Math.cos(p) * Math.sin(l) };
}

/** Solar elevation angle (deg) at a lat/lon point. Positive = sun above the horizon. */
export function solarElevation(lat, lon, ms) {
  const s = solarPosition(ms);
  const u = latLonToUnit(lat, lon);
  const dot = u.x * s.dir.x + u.y * s.dir.y + u.z * s.dir.z;
  return Math.asin(Math.max(-1, Math.min(1, dot))) * RAD;
}
