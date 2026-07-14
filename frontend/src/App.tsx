import { useState, useEffect } from "react";
import { ExternalLink, Wifi, WifiOff } from "lucide-react";
import { UploadZone } from "./components/UploadZone";
import { MetricsBar } from "./components/MetricsBar";
import { ImagePanel } from "./components/ImagePanel";
import { reconstruct, checkHealth, type ReconstructResult } from "./api/client";

export default function App() {
  const [result, setResult] = useState<ReconstructResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  useEffect(() => {
    checkHealth().then(setApiOnline);
  }, []);

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    setFilename(file.name);
    try {
      const data = await reconstruct(file);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">E2E-VarNet MRI Reconstruction</h1>
            <p className="text-xs text-gray-500">
              Sriram et al., MICCAI 2020 — reproduction + uncertainty estimation
            </p>
          </div>
          <div className="flex items-center gap-3">
            {apiOnline !== null && (
              <span className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full
                ${apiOnline
                  ? "bg-green-100 text-green-700"
                  : "bg-red-100 text-red-700"}`}>
                {apiOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                {apiOnline ? "API online" : "API offline"}
              </span>
            )}
            <a
              href="https://github.com/alyrraza/e2e-varnet-mri-reconstruction"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
              GitHub
            </a>
            <a
              href="https://huggingface.co/alyrraza/e2e-varnet-mri-reconstruction"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              <ExternalLink className="w-4 h-4" /> Weights
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        {/* Info banner */}
        <div className="bg-blue-50 border border-blue-200 rounded-xl px-6 py-4 text-sm text-blue-800">
          <strong>How it works:</strong> Upload a fastMRI single-coil knee{" "}
          <code className="bg-blue-100 px-1 rounded">.h5</code> file. The model
          (E2E-VarNet T=4, 4x acceleration) reconstructs the image from 25% of
          k-space and returns a checkpoint-ensemble uncertainty score that flags
          out-of-distribution scans. Shift detection: <strong>1.54x</strong> higher
          uncertainty on OOD brain data, bootstrap 95% CI [1.35x, 1.74x].
        </div>

        {/* Upload */}
        <section>
          <UploadZone onFile={handleFile} loading={loading} />
        </section>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-6 py-4 text-red-700 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-800">Results</h2>
              {filename && (
                <span className="text-sm text-gray-500 font-mono">{filename}</span>
              )}
            </div>

            {/* Metrics */}
            <MetricsBar result={result} />

            {/* Images */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ImagePanel
                title="Reconstruction"
                subtitle="E2E-VarNet T=4 output"
                b64={result.reconstruction_b64}
                border="border-blue-200"
              />
              <ImagePanel
                title="Uncertainty Map"
                subtitle="Per-pixel std across K=2 checkpoints"
                b64={result.uncertainty_map_b64}
                border={result.flagged_for_review ? "border-red-300" : "border-green-200"}
              />
            </div>
          </section>
        )}

        {/* Results table */}
        <section className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100">
            <h2 className="font-semibold text-gray-800">Official Results (Locked)</h2>
            <p className="text-xs text-gray-500 mt-1">
              Volume-level mean SSIM. Single-coil knee 4x acceleration.
              Not comparable to paper's 0.930 (multi-coil).
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-600 text-xs uppercase tracking-wide">
                <tr>
                  <th className="px-6 py-3 text-left">Model</th>
                  <th className="px-6 py-3 text-right">Knee SSIM</th>
                  <th className="px-6 py-3 text-right">Brain SSIM (OOD)</th>
                  <th className="px-6 py-3 text-right">OOD Drop</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {[
                  { model: "Zero-filled", knee: "0.7453", brain: "0.4153", drop: "--" },
                  { model: "T=4 (50 epochs)", knee: "0.7594", brain: "0.6622", drop: "-0.097" },
                  { model: "T=6 (25 epochs)", knee: "0.7606", brain: "0.6705", drop: "-0.090" },
                  { model: "T=8 (24 epochs)", knee: "0.7607", brain: "0.6773", drop: "-0.083" },
                ].map((row) => (
                  <tr key={row.model} className="hover:bg-gray-50">
                    <td className="px-6 py-3 font-medium text-gray-800">{row.model}</td>
                    <td className="px-6 py-3 text-right tabular-nums">{row.knee}</td>
                    <td className="px-6 py-3 text-right tabular-nums">{row.brain}</td>
                    <td className="px-6 py-3 text-right tabular-nums text-red-600">{row.drop}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-xs text-gray-400 pb-8">
          E2E-VarNet reproduction by{" "}
          <a href="https://github.com/alyrraza" className="hover:underline">alyrraza</a>.{" "}
          Base: Sriram et al., MICCAI 2020.
          All numbers are volume-level means, independently stress-tested.
        </footer>
      </main>
    </div>
  );
}
