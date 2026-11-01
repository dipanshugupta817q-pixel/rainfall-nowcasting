type MetricCardProps = {
  label: string;
  value: string;
  unit: string;
  status: string;
};

export default function MetricCard({
  label,
  value,
  unit,
  status,
}: MetricCardProps) {
  return (
    <div className="min-w-0 rounded-2xl border border-white/10 bg-slate-900 p-5 transition hover:border-white/20">
      <p className="text-xs font-medium tracking-widest text-slate-500">
        {label}
      </p>
      <div className="mt-4 flex items-baseline gap-2">
        <span className="text-3xl font-semibold tracking-tight">{value}</span>
        <span className="text-sm text-slate-400">{unit}</span>
      </div>
      <p className="mt-2 text-xs text-slate-600">{status}</p>
    </div>
  );
}
