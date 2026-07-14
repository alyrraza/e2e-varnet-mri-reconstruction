interface Props {
  title: string;
  subtitle: string;
  b64: string;
  border?: string;
}

export function ImagePanel({ title, subtitle, b64, border = "border-gray-200" }: Props) {
  return (
    <div className={`border ${border} rounded-xl overflow-hidden`}>
      <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
        <p className="font-semibold text-sm text-gray-800">{title}</p>
        <p className="text-xs text-gray-500">{subtitle}</p>
      </div>
      <div className="bg-black flex items-center justify-center p-2">
        <img
          src={`data:image/png;base64,${b64}`}
          alt={title}
          className="max-w-full max-h-72 object-contain"
          style={{ imageRendering: "pixelated" }}
        />
      </div>
    </div>
  );
}
