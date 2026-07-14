import { AlertTriangle, CheckCircle } from "lucide-react";
import type { ReconstructResult } from "../api/client";

interface Props {
  result: ReconstructResult;
}

export function MetricsBar({ result }: Props) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard
        label="SSIM"
        value={result.ssim.toFixed(4)}
        sub="vs zero-filled"
        color="blue"
      />
      <MetricCard
        label="PSNR"
        value={`${result.psnr.toFixed(1)} dB`}
        sub="vs zero-filled"
        color="blue"
      />
      <MetricCard
        label="Uncertainty"
        value={result.uncertainty_scalar.toFixed(5)}
        sub="normalised scalar"
        color={result.flagged_for_review ? "red" : "green"}
      />
      <MetricCard
        label="Inference"
        value={`${result.inference_time_ms.toFixed(0)} ms`}
        sub={`model: ${result.model_variant}`}
        color="gray"
      />

      <div className="col-span-2 md:col-span-4">
        {result.flagged_for_review ? (
          <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-red-700">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <span className="text-sm font-medium">
              Drift flag: uncertainty scalar exceeds threshold. This scan may be
              out-of-distribution (e.g. different anatomy or scanner than training data).
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg px-4 py-3 text-green-700">
            <CheckCircle className="w-5 h-5 shrink-0" />
            <span className="text-sm font-medium">
              No drift detected. Uncertainty scalar is within the in-distribution range.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({
  label, value, sub, color,
}: {
  label: string; value: string; sub: string;
  color: "blue" | "green" | "red" | "gray";
}) {
  const colors = {
    blue: "text-blue-700 bg-blue-50 border-blue-200",
    green: "text-green-700 bg-green-50 border-green-200",
    red: "text-red-700 bg-red-50 border-red-200",
    gray: "text-gray-700 bg-gray-50 border-gray-200",
  };

  return (
    <div className={`border rounded-lg px-4 py-3 ${colors[color]}`}>
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      <p className="text-xs opacity-60 mt-0.5">{sub}</p>
    </div>
  );
}
