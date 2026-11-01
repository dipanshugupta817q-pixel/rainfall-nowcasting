import type { ForecastData, ForecastFrame } from "@/lib/types";

export async function loadForecast(
  latitude?: number,
  longitude?: number,
): Promise<ForecastData> {
  const params = new URLSearchParams();

  if (latitude !== undefined && longitude !== undefined) {
    params.set("lat", latitude.toFixed(6));
    params.set("lon", longitude.toFixed(6));
  }

  const query = params.toString();
  const url = query
    ? `/api/forecast?${query}`
    : "/api/forecast";

  const response = await fetch(url, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Failed to load forecast (${response.status})`);
  }

  const data = (await response.json()) as ForecastData;

  if (!data.frames?.length) {
    throw new Error("Forecast response contains no forecast frames.");
  }

  return data;
}

export function formatHorizon(minutes: number): string {
  if (minutes === 0) return "Now";
  if (minutes < 60) return `+${minutes} min`;

  const hours = minutes / 60;

  return Number.isInteger(hours)
    ? `+${hours} hr`
    : `+${hours.toFixed(1)} hr`;
}

export function formatNumber(value: number, decimals = 2): string {
  return Number.isFinite(value) ? value.toFixed(decimals) : "—";
}

export function getIntensityLabel(value: number): string {
  if (!Number.isFinite(value)) return "UNKNOWN";
  if (value < 0.1) return "NO RAIN";
  if (value < 2.5) return "LIGHT";
  if (value < 7.5) return "MODERATE";
  if (value < 15) return "HEAVY";
  if (value < 30) return "VERY HEAVY";
  return "EXTREME";
}

export function getIntensityDot(value: number): string {
  if (value < 0.1) return "bg-slate-700";
  if (value < 2.5) return "bg-cyan-400";
  if (value < 7.5) return "bg-blue-400";
  if (value < 15) return "bg-violet-400";
  if (value < 30) return "bg-orange-400";
  return "bg-red-400";
}

export function getPeakForecast(
  frames: ForecastFrame[],
): ForecastFrame | null {
  if (!frames.length) return null;

  return frames.reduce(
    (highest, frame) =>
      frame.max > highest.max ? frame : highest,
    frames[0],
  );
}

export function getAverageForecast(frames: ForecastFrame[]): number {
  if (!frames.length) return 0;

  return (
    frames.reduce((sum, frame) => sum + frame.mean, 0) /
    frames.length
  );
}
