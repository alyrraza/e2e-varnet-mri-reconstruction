const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface ReconstructResult {
  ssim: number;
  psnr: number;
  uncertainty_scalar: number;
  flagged_for_review: boolean;
  reconstruction_b64: string;
  uncertainty_map_b64: string;
  inference_time_ms: number;
  model_variant: string;
}

export async function reconstruct(file: File): Promise<ReconstructResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/reconstruct`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
