import { useRef, useState, type DragEvent } from "react";
import { Upload, FileX } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
}

export function UploadZone({ onFile, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFile(file: File) {
    setError(null);
    if (!file.name.endsWith(".h5")) {
      setError("Only fastMRI .h5 files are supported.");
      return;
    }
    onFile(file);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !loading && inputRef.current?.click()}
      className={`
        border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
        transition-colors duration-200 select-none
        ${dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"}
        ${loading ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".h5"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />

      {loading ? (
        <div className="flex flex-col items-center gap-3 text-blue-600">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="font-medium">Reconstructing...</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 text-gray-500">
          <Upload className="w-10 h-10 text-blue-400" />
          <p className="font-medium text-gray-700">
            Drop a fastMRI <code className="bg-gray-100 px-1 rounded">.h5</code> file here
          </p>
          <p className="text-sm">or click to browse</p>
        </div>
      )}

      {error && (
        <div className="mt-4 flex items-center gap-2 text-red-600 justify-center text-sm">
          <FileX className="w-4 h-4" /> {error}
        </div>
      )}
    </div>
  );
}
