type LayerProps = {
  name: string;
  active?: boolean;
  comingSoon?: boolean;
};

function Layer({
  name,
  active = false,
  comingSoon = false,
}: LayerProps) {
  return (
    <button
      type="button"
      disabled={comingSoon}
      className={`rounded-lg border px-4 py-2 text-sm transition ${
        active
          ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-300 hover:bg-cyan-400/15"
          : comingSoon
            ? "cursor-not-allowed border-white/10 text-slate-600"
            : "border-white/10 text-slate-400 hover:bg-white/5 hover:text-slate-200"
      }`}
    >
      {name}
      {comingSoon && (
        <span className="ml-2 text-[9px] tracking-wide text-slate-600">
          SOON
        </span>
      )}
    </button>
  );
}

export default function DataLayers() {
  return (
    <div className="mt-6 w-full min-w-0 rounded-2xl border border-white/10 bg-slate-900 p-5">
      <div className="mb-4">
        <h3 className="font-medium">Data Layers</h3>
        <p className="mt-1 text-xs text-slate-500">
          Visualize different weather data sources
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <Layer name="Rainfall" active />
        <Layer name="Forecast" active />
        <Layer name="Wind" comingSoon />
        <Layer name="Temperature" comingSoon />
        <Layer name="Satellite" comingSoon />
        <Layer name="Radar" comingSoon />
      </div>
    </div>
  );
}
