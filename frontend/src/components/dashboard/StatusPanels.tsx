type LocationStatusProps = {
  locationLabel: string;
  loading?: boolean;
};

export function LocationStatus({
  locationLabel,
  loading = false,
}: LocationStatusProps) {
  return (
    <div className="mt-6 w-full rounded-2xl border border-white/10 bg-slate-900/60 px-5 py-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-slate-300">
            Forecast Location
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {locationLabel}
          </p>
        </div>

        <span className="rounded-lg border border-cyan-400/20 bg-cyan-400/5 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-cyan-300">
          {loading ? "Updating" : "Active"}
        </span>
      </div>
    </div>
  );
}

export function SystemStatus() {
  return (
    <div className="mt-6 flex w-full min-w-0 flex-wrap items-center justify-between gap-4 rounded-2xl border border-white/10 bg-slate-900/60 px-5 py-4">
      <div className="flex items-center gap-3">
        <span className="h-2 w-2 rounded-full bg-amber-400" />
        <span className="text-xs text-slate-400">
          RainCast AI — Forecast environment
        </span>
      </div>

      <div className="flex gap-5 text-xs text-slate-600">
        <span>IMERG</span>
        <span>ERA5</span>
        <span>Forecast Engine</span>
      </div>
    </div>
  );
}
