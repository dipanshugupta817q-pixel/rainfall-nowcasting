import type { ForecastData } from "@/lib/types";

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[9px] font-medium tracking-widest text-slate-600">{label}</p>
      <p className="mt-1 truncate text-xs font-medium text-slate-300">{value}</p>
    </div>
  );
}

export default function ModelInfo({
  forecast,
}: {
  forecast: ForecastData | null;
}) {
  const horizon = forecast?.frames.length
    ? Math.max(...forecast.frames.map((frame) => frame.horizon_minutes))
    : 360;

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900 p-5">
      <div>
        <h3 className="font-medium">Forecast Information</h3>
        <p className="mt-1 text-xs text-slate-500">
          Coverage and resolution of the current forecast
        </p>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4">
        <InfoItem
          label="GRID"
          value={
            forecast
              ? `${forecast.grid.height} × ${forecast.grid.width}`
              : "—"
          }
        />
        <InfoItem label="UNIT" value="mm/hr" />
        <InfoItem
          label="LATITUDE"
          value={
            forecast
              ? `${forecast.grid.lat_min.toFixed(2)}° – ${forecast.grid.lat_max.toFixed(2)}°`
              : "—"
          }
        />
        <InfoItem
          label="LONGITUDE"
          value={
            forecast
              ? `${forecast.grid.lon_min.toFixed(2)}° – ${forecast.grid.lon_max.toFixed(2)}°`
              : "—"
          }
        />
        <InfoItem label="FORECAST HORIZON" value={`${Math.round(horizon / 60)} hours`} />
        <InfoItem
          label="TIME STEPS"
          value={forecast ? String(forecast.frames.length) : "—"}
        />
      </div>
    </div>
  );
}
