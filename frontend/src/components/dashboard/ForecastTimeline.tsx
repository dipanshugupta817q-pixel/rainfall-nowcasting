import type { ForecastFrame } from "@/lib/types";
import { formatHorizon, formatNumber, getIntensityDot } from "@/lib/forecast";

type ForecastTimelineProps = {
  frames: ForecastFrame[];
};

export default function ForecastTimeline({
  frames,
}: ForecastTimelineProps) {
  return (
    <div className="mt-6 w-full min-w-0 rounded-2xl border border-white/10 bg-slate-900 p-5">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="font-medium">Forecast Timeline</h3>
          <p className="mt-1 text-xs text-slate-500">
            Rainfall intensity over the next 6 hours
          </p>
        </div>
        <div className="text-xs text-slate-500">mm/hr</div>
      </div>

      <div className="relative overflow-x-auto pb-2">
        <div className="min-w-[720px]">
          <div className="relative">
            <div className="absolute left-0 right-0 top-3 h-px bg-white/10" />

            <div className="relative flex justify-between">
              {frames.map((frame) => (
                <div
                  key={frame.horizon_minutes}
                  className="flex flex-col items-center"
                >
                  <div
                    className={`z-10 h-6 w-6 rounded-full border-4 border-slate-900 ${getIntensityDot(
                      frame.mean,
                    )}`}
                  />

                  <div className="mt-3 text-xs text-slate-500">
                    {formatHorizon(frame.horizon_minutes)}
                  </div>

                  <div className="mt-1 text-xs font-medium text-slate-300">
                    {formatNumber(frame.mean)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
