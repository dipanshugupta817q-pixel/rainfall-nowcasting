"use client";

import { useMemo, useState } from "react";
import type { ForecastFrame } from "@/lib/types";
import { formatHorizon, formatNumber, getIntensityLabel } from "@/lib/forecast";

type ForecastChartProps = {
  frames: ForecastFrame[];
};

const WIDTH = 1000;
const HEIGHT = 300;
const PAD_X = 54;
const PAD_Y = 30;
const CHART_W = WIDTH - PAD_X * 2;
const CHART_H = HEIGHT - PAD_Y * 2;

export default function ForecastChart({ frames }: ForecastChartProps) {
  const [hovered, setHovered] = useState<number | null>(null);

  const chart = useMemo(() => {
    if (!frames.length) return null;

    const maxValue = Math.max(
      1,
      ...frames.map((frame) => frame.mean),
    );
    const yMax = Math.ceil(maxValue * 1.2 * 10) / 10;

    const points = frames.map((frame, index) => {
      const x =
        frames.length === 1
          ? PAD_X + CHART_W / 2
          : PAD_X + (index / (frames.length - 1)) * CHART_W;

      const y =
        PAD_Y + CHART_H - (frame.mean / yMax) * CHART_H;

      return { x, y, frame };
    });

    const line = points
      .map((point) => `${point.x},${point.y}`)
      .join(" ");

    const area = [
      `${PAD_X},${PAD_Y + CHART_H}`,
      ...points.map((point) => `${point.x},${point.y}`),
      `${PAD_X + CHART_W},${PAD_Y + CHART_H}`,
    ].join(" ");

    return { points, line, area, yMax };
  }, [frames]);

  if (!chart) {
    return (
      <div className="flex h-[300px] items-center justify-center rounded-xl border border-white/5 bg-black/10 text-sm text-slate-600">
        No forecast data available.
      </div>
    );
  }

  const activeIndex = hovered ?? Math.min(1, frames.length - 1);
  const active = chart.points[activeIndex];

  return (
    <div className="w-full overflow-hidden rounded-xl border border-white/5 bg-[#0d1514] p-3">
      <div className="mb-3 flex items-center justify-between px-2">
        <div>
          <p className="text-xs font-medium tracking-wide text-slate-300">
            Mean Rainfall Intensity
          </p>
          <p className="mt-1 text-[10px] text-slate-600">
            Rainfall forecast - 12 time steps
          </p>
        </div>

        {active && (
          <div className="text-right">
            <p className="text-sm font-semibold text-cyan-300">
              {formatNumber(active.frame.mean)} mm/hr
            </p>
            <p className="text-[10px] text-slate-600">
              {formatHorizon(active.frame.horizon_minutes)} â€¢{" "}
              {getIntensityLabel(active.frame.mean)}
            </p>
          </div>
        )}
      </div>

      <div className="relative w-full">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="block h-auto w-full"
          role="img"
          aria-label="Forecast rainfall intensity chart"
        >
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
            const y = PAD_Y + CHART_H * (1 - fraction);
            const value = chart.yMax * fraction;

            return (
              <g key={fraction}>
                <line
                  x1={PAD_X}
                  x2={PAD_X + CHART_W}
                  y1={y}
                  y2={y}
                  stroke="rgba(255,255,255,0.07)"
                  strokeWidth="1"
                />
                <text
                  x={PAD_X - 10}
                  y={y + 4}
                  textAnchor="end"
                  fill="#64706d"
                  fontSize="11"
                >
                  {value.toFixed(1)}
                </text>
              </g>
            );
          })}

          <polygon
            points={chart.area}
            fill="rgba(56,217,176,0.08)"
          />

          <polyline
            points={chart.line}
            fill="none"
            stroke="#38D9B0"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {chart.points.map((point, index) => {
            const selected = index === activeIndex;

            return (
              <g key={point.frame.horizon_minutes}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={selected ? 6 : 4}
                  fill={selected ? "#38D9B0" : "#17201F"}
                  stroke="#38D9B0"
                  strokeWidth={selected ? 3 : 2}
                />

                <rect
                  x={point.x - 24}
                  y={PAD_Y}
                  width="48"
                  height={CHART_H}
                  fill="transparent"
                  onMouseEnter={() => setHovered(index)}
                  onMouseLeave={() => setHovered(null)}
                />

                {(index === 0 || index === frames.length - 1) && (
                  <text
                    x={point.x}
                    y={HEIGHT - 5}
                    textAnchor="middle"
                    fill="#64706d"
                    fontSize="11"
                  >
                    {formatHorizon(point.frame.horizon_minutes)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
