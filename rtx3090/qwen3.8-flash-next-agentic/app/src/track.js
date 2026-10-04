// Ground-track math shared by the renderer and the tests.
// Frame convention matches src/solar.js and the equirectangular textures:
//   x = cos(lat)cos(lon), y = sin(lat), z = -cos(lat)sin(lon)
// u = 0 is longitude -180, v = 0 is latitude +90.

const DEG = Math.PI / 180;

export function latLonToVec3(lat, lon, r) {
  const p = lat * DEG, l = lon * DEG;
  return { x: r * Math.cos(p) * Math.cos(l), y: r * Math.sin(p), z: -r * Math.cos(p) * Math.sin(l) };
}

/** Shortest signed longitude delta in degrees. */
export function lonDelta(a, b) {
  let d = b - a;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return d;
}

/** Degrees per second of lat/lon motion between two samples. */
export function groundTrackRate(prev, curr) {
  const dt = curr.timestamp - prev.timestamp || 1;
  return {
    dlat: (curr.latitude - prev.latitude) / dt,
    dlon: lonDelta(prev.longitude, curr.longitude) / dt,
    dt,
  };
}

/** Great-circle distance (km) between two lat/lon points. */
export function greatCircleKm(a, b) {
  const R = 6371;
  const p1 = a.latitude * DEG, p2 = b.latitude * DEG;
  const dp = p2 - p1, dl = lonDelta(a.longitude, b.longitude) * DEG;
  const h =
    Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Derived ground-track speed in km/h from two samples. */
export function derivedSpeedKmH(prev, curr) {
  const dt = curr.timestamp - prev.timestamp;
  if (!dt) return 0;
  return (greatCircleKm(prev, curr) / dt) * 3600;
}

/** Dead-reckoned position at time t (epoch ms) from the last sample + rate.
 *  API timestamps are epoch seconds; convert before differencing. */
export function propagate(curr, rate, tMs) {
  const dt = (tMs - curr.timestamp * 1000) / 1000;
  const lat = curr.latitude + rate.dlat * dt;
  const lon = curr.longitude + rate.dlon * dt;
  return { lat, lon: ((lon + 540) % 360) - 180 };
}
