import type { ForecastFrame } from "@/lib/types";
import { formatHorizon, formatNumber, getIntensityDot } from "@/lib/forecast";

type ForecastPanelProps = {
  frames: ForecastFrame[];
};

export default function ForecastPanel({ frames }: ForecastPanelProps) {
  return (
    <div className="min-w-0 w-full overflow-hidden rounded-2xl border border-white/10 bg-slate-900">
      <div className="border-b border-white/10 px-5 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">6-Hour Nowcast</h3>
            <p className="mt-1 text-xs text-slate-500">
              Predicted rainfall intensity
            </p>
          </div>
          <div className="rounded-md bg-cyan-400/10 px-2 py-1 text-[10px] font-medium text-cyan-300">
            NOWCAST
          </div>
        </div>
      </div>

      <div className="divide-y divide-white/5">
        {frames.map((frame, index) => (
          <div
            key={frame.horizon_minutes}
            className={`flex items-center justify-between px-5 py-3.5 transition ${
              index === 1 ? "bg-cyan-400/5" : ""
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className={`h-2 w-2 rounded-full ${getIntensityDot(frame.mean)}`}
              />
              <span className="text-sm text-slate-400">
                {formatHorizon(frame.horizon_minutes)}
              </span>
            </div>

            <div className="text-right">
              <span className="font-medium text-white">
                {formatNumber(frame.mean)}
              </span>
              <span className="ml-1 text-xs text-slate-500">mm/hr</span>
            </div>
          </div>
        ))}

        {!frames.length && (
          <div className="px-5 py-8 text-center text-xs text-slate-600">
            No forecast frames available.
          </div>
        )}
      </div>
    </div>
  );
}
