type SmallMetricProps = {
  label: string;
  value: string;
  detail: string;
};

export default function SmallMetric({
  label,
  value,
  detail,
}: SmallMetricProps) {
  return (
    <div className="min-w-0 rounded-2xl border border-white/10 bg-slate-900/70 p-4">
      <p className="text-[10px] font-medium tracking-widest text-slate-600">
        {label}
      </p>
      <p className="mt-2 text-lg font-semibold text-slate-200">{value}</p>
      <p className="mt-1 text-[10px] text-slate-600">{detail}</p>
    </div>
  );
}
