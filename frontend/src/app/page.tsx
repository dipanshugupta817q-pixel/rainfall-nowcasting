"use client";

import { useEffect, useMemo, useState } from "react";
import type { LocationTarget } from "@/components/LocationSearch";
import RainfallMap from "@/components/RainfallMap";
import MetricCard from "@/components/dashboard/MetricCard";
import SmallMetric from "@/components/dashboard/SmallMetric";
import ForecastPanel from "@/components/dashboard/ForecastPanel";
import ForecastTimeline from "@/components/dashboard/ForecastTimeline";
import ForecastChart from "@/components/dashboard/ForecastChart";

import DataLayers from "@/components/dashboard/DataLayers";
import {
  LocationStatus,
  SystemStatus,
} from "@/components/dashboard/StatusPanels";
import {
  formatHorizon,
  formatNumber,
  getAverageForecast,
  getIntensityLabel,
  getPeakForecast,
  loadForecast,
} from "@/lib/forecast";
import type { ForecastData } from "@/lib/types";

export default function Home() {
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLocationLabel, setSelectedLocationLabel] = useState("Forecast coverage area");
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;

    loadForecast()
      .then((data) => {
        if (!cancelled) {
          setForecast(data);
          setSelectedFrameIndex(0);
        }
      })
      .catch((err) => {
        console.error("Dashboard forecast loading error:", err);
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load forecast data.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });return () => {
      cancelled = true;
    };
  }, []);


async function handleLocationSelect(target: LocationTarget) {
    setSelectedLocationLabel(target.label ?? "Selected location");
    setSelectedFrameIndex(0);
    setLoading(true);
    setError(null);

    try {
      const data = await loadForecast(
        target.latitude,
        target.longitude,
      );

      setForecast(data);

      if (data.location?.label) {
        setSelectedLocationLabel(data.location.label);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load forecast for the selected location.",
      );
    } finally {
      setLoading(false);
    }
  }

  const frames = forecast?.frames ?? [];
  const safeFrameIndex = Math.min(
    selectedFrameIndex,
    Math.max(frames.length - 1, 0),
  );
  const currentFrame = frames[safeFrameIndex];

  const thirtyMinuteFrame =
    frames.find((frame) => frame.horizon_minutes === 30) ?? frames[1];

  const oneHourFrame =
    frames.find((frame) => frame.horizon_minutes === 60) ?? frames[2];

  const peakForecast = useMemo(
    () => getPeakForecast(frames),
    [frames],
  );

  const averageForecast = useMemo(
    () => getAverageForecast(frames),
    [frames],
  );

  const maxForecastValue = peakForecast?.max ?? 0;

  return (
    <main className="min-h-screen w-full overflow-x-hidden bg-[#0B1110] text-[#F1F5F3]">
      <header className="border-b border-white/10 bg-[#0B1110]/95">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#38D9B0] font-bold text-[#0B1110]">
              R
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                RainCast AI
              </h1>
              <p className="text-xs text-[#64706d]">
                Rainfall Nowcasting System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <span className="h-2 w-2 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.7)]" />
              LIVE
            </div>
            <div className="hidden text-sm text-slate-300 sm:block">
              Meerut, India
            </div>
            <button
              type="button"
              aria-label="Settings"
              className="rounded-lg border border-white/10 px-3 py-2 text-slate-400 transition hover:bg-white/5 hover:text-white"
            >
              âš™
            </button>
          </div>
        </div>
      </header>

      <section className="mx-auto w-full max-w-7xl px-6 py-8">
        <div className="mb-8">
          <p className="mb-2 text-sm font-medium tracking-widest text-[#38D9B0]">
            AI-POWERED WEATHER INTELLIGENCE
          </p>
          <h2 className="text-4xl font-semibold tracking-tight md:text-5xl">
            Heavy Rainfall
            <br />
            <span className="text-slate-400">Nowcasting</span>
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-400">
            Short-term rainfall prediction for the next 1â€“6 hours, powered by
            atmospheric and precipitation data.
          </p>
        </div>

        {loading && (
          <div className="mb-6 flex items-center gap-3 rounded-xl border border-white/10 bg-[#111918] px-5 py-4">
            <span className="h-2 w-2 animate-pulse rounded-full bg-[#38D9B0]" />
            <span className="text-sm text-slate-400">
              Loading forecast data...
            </span>
          </div>
        )}

        {error && (
          <div className="mb-6 rounded-xl border border-red-400/20 bg-red-400/5 px-5 py-4">
            <p className="text-sm font-medium text-red-300">
              Forecast data unavailable
            </p>
            <p className="mt-1 text-xs text-red-300/60">{error}</p>
          </div>
        )}

        <div className="grid w-full min-w-0 gap-4 md:grid-cols-3">
          <MetricCard
            label="CURRENT RAINFALL"
            value={currentFrame ? formatNumber(currentFrame.mean) : "â€”"}
            unit="mm/hr"
            status={
              currentFrame ? "CURRENT FORECAST" : "WAITING FOR DATA"
            }
          />

          <MetricCard
            label="+30 MIN FORECAST"
            value={
              thirtyMinuteFrame
                ? formatNumber(thirtyMinuteFrame.mean)
                : "â€”"
            }
            unit="mm/hr"
            status={
              thirtyMinuteFrame
                ? getIntensityLabel(thirtyMinuteFrame.mean)
                : "WAITING FOR DATA"
            }
          />

          <MetricCard
            label="+1 HOUR FORECAST"
            value={
              oneHourFrame ? formatNumber(oneHourFrame.mean) : "â€”"
            }
            unit="mm/hr"
            status={
              oneHourFrame
                ? getIntensityLabel(oneHourFrame.mean)
                : "WAITING FOR DATA"
            }
          />
        </div>

        <div className="mt-4 grid w-full min-w-0 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <SmallMetric
            label="PEAK GRID INTENSITY"
            value={
              peakForecast
                ? `${formatNumber(peakForecast.max)} mm/hr`
                : "â€”"
            }
            detail={
              peakForecast
                ? formatHorizon(peakForecast.horizon_minutes)
                : "No forecast"
            }
          />

          <SmallMetric
            label="FORECAST AVERAGE"
            value={
              frames.length
                ? `${formatNumber(averageForecast)} mm/hr`
                : "â€”"
            }
            detail={`${frames.length} forecast frames`}
          />

          <SmallMetric
            label="FORECAST GRID"
            value={
              forecast
                ? `${forecast.grid.height} Ã— ${forecast.grid.width}`
                : "â€”"
            }
            detail="Spatial forecast resolution"
          />
        </div>

        <div className="mt-6 grid w-full min-w-0 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="flex min-w-0 w-full max-w-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#111918] shadow-2xl shadow-black/20">
            <div className="flex shrink-0 items-center justify-between border-b border-white/10 bg-[#111918] px-5 py-4">
              <div>
                <h3 className="font-medium text-white">Rainfall Map</h3>
                <p className="mt-1 text-xs text-slate-500">
                  {selectedLocationLabel} • Short-term rainfall forecast
                </p>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1.5 text-xs text-cyan-300"
                >
                  Rainfall
                </button>
                <button
                  type="button"
                  className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-400 transition hover:bg-white/5"
                >
                  Layers
                </button>
              </div>
            </div>

            <div
              className="relative block h-[520px] w-full min-w-0 max-w-full flex-none overflow-hidden"
              style={{ width: "100%", minWidth: 0, maxWidth: "100%" }}
            >
              <RainfallMap
                forecast={forecast}
                frameIndex={safeFrameIndex}
                loading={loading}
                onFrameChange={setSelectedFrameIndex}
                onLocationSelect={handleLocationSelect}
/>
            </div>
          </div>

          <ForecastPanel frames={frames} />
        </div>

        <ForecastChart frames={frames} />

        <ForecastTimeline frames={frames} />

        <div className="mt-6 grid w-full min-w-0 gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-[#111918] p-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium">Peak Forecast</h3>
                <p className="mt-1 text-xs text-slate-500">
                  Highest grid-cell intensity across forecast
                </p>
              </div>

              <div
                className={`rounded-lg px-3 py-1.5 text-[10px] font-semibold ${
                  maxForecastValue >= 15
                    ? "bg-red-400/10 text-red-300"
                    : maxForecastValue >= 7.5
                      ? "bg-orange-400/10 text-orange-300"
                      : "bg-cyan-400/10 text-cyan-300"
                }`}
              >
                {getIntensityLabel(maxForecastValue)}
              </div>
            </div>

            <div className="mt-5 flex items-end justify-between">
              <div>
                <span className="text-4xl font-semibold tracking-tight">
                  {formatNumber(maxForecastValue)}
                </span>
                <span className="ml-2 text-sm text-slate-500">mm/hr</span>
              </div>

              <div className="text-right">
                <p className="text-xs text-slate-500">Forecast horizon</p>
                <p className="mt-1 text-sm font-medium text-slate-300">
                  {peakForecast
                    ? formatHorizon(peakForecast.horizon_minutes)
                    : "â€”"}
                </p>
              </div>
            </div>
          </div>


        </div>

        <DataLayers />
        <LocationStatus locationLabel={selectedLocationLabel} loading={loading} />
        <SystemStatus />
      </section>
    </main>
  );
}



