"use client";

import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { PREDICTION_MODEL_OPTIONS, predictScoliosisImage } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { PredictionResult } from "@/components/prediction/PredictionResult";
import { getExportVertebraColor, PROJECT_DISCLAIMER } from "@/lib/constants";
import {
  formatConfidenceDecimal,
  getImageSize,
  getLayerLabel,
  getOverlayLayers,
  getPolygonPoints,
  getSegmentColors,
  getSegmentKey,
  getSegmentLabel,
  type OverlayLayer,
} from "@/lib/prediction-results";
import type {
  PredictionModelKey,
  PredictionResultsByModel,
} from "@/types/prediction";

type SampleImage = {
  alt: string;
  label: string;
  src: string;
};

type UploadCardProps = {
  sampleImages?: SampleImage[];
};

type RequestState =
  | { status: "idle"; error: null }
  | { status: "loading"; error: null }
  | { status: "error"; error: string };

type PanelTab = "setup" | "results";
type DownloadKind = "color-mask" | "binary-selection" | "overlay";

const INITIAL_VISIBLE_LAYERS = {
  binary: true,
  multiclass: true,
};

const DOWNLOAD_VISIBLE_LAYERS = {
  binary: true,
  multiclass: true,
};

function responseHasLayer(
  response: PredictionResultsByModel[PredictionModelKey],
  layer: "binary" | "multiclass",
) {
  if (!response?.results) {
    return false;
  }

  if (response.results[layer]?.segments?.length) {
    return true;
  }

  return Object.values(response.results).some(
    (result) => result.type === layer && Boolean(result.segments?.length),
  );
}

function getFilenameFromSrc(src: string) {
  return decodeURIComponent(src.split("/").pop() ?? "sample-xray.jpg");
}

function getDownloadBaseName(file: File | null) {
  return (file?.name ?? "spine-prediction")
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9-_]+/gi, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
}

function loadImageElement(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to prepare image download."));
    image.src = src;
  });
}

function downloadCanvas(canvas: HTMLCanvasElement, filename: string) {
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function fillSegmentPath(
  context: CanvasRenderingContext2D,
  segment: OverlayLayer["segments"][number],
  fillStyle: string,
) {
  const polygon = getPolygonPoints(segment);

  if (polygon.length) {
    context.beginPath();
    context.moveTo(polygon[0][0], polygon[0][1]);
    polygon.slice(1).forEach(([x, y]) => context.lineTo(x, y));
    context.closePath();
    context.fillStyle = fillStyle;
    context.fill();
    return;
  }

  const bbox = segment.bbox;

  if (bbox?.length === 4) {
    context.fillStyle = fillStyle;
    context.fillRect(
      bbox[0],
      bbox[1],
      Math.max(0, bbox[2] - bbox[0]),
      Math.max(0, bbox[3] - bbox[1]),
    );
  }
}

function drawOverlayLabels(
  context: CanvasRenderingContext2D,
  layer: OverlayLayer,
  showBoundingBoxes: boolean,
  showBoxLabels: boolean,
  showBoxConfidence: boolean,
  hiddenSegments: Record<string, boolean>,
) {
  if (layer.key !== "multiclass" || !showBoundingBoxes) {
    return;
  }

  layer.segments.forEach((segment, index) => {
    if (hiddenSegments[getSegmentKey(layer.key, segment, index)]) {
      return;
    }

    const bbox = segment.bbox;

    if (!bbox || bbox.length !== 4) {
      return;
    }

    const colors = getSegmentColors(segment, layer.key);
    const confidence = formatConfidenceDecimal(segment);
    const label = [
      showBoxLabels ? getSegmentLabel(segment) : null,
      showBoxConfidence ? confidence : null,
    ]
      .filter(Boolean)
      .join(" ");

    if (!label) {
      return;
    }

    context.save();
    context.strokeStyle = colors.border;
    context.lineWidth = 3;
    context.strokeRect(
      bbox[0],
      bbox[1],
      Math.max(0, bbox[2] - bbox[0]),
      Math.max(0, bbox[3] - bbox[1]),
    );
    context.setLineDash([]);
    context.font = "700 22px Arial, sans-serif";
    context.lineWidth = 5;
    context.strokeStyle = "#0d1620";
    context.fillStyle = "#ffffff";
    context.strokeText(label, bbox[0], Math.max(24, bbox[1] - 8));
    context.fillText(label, bbox[0], Math.max(24, bbox[1] - 8));
    context.restore();
  });
}

function ToggleButton({
  checked,
  children,
  onChange,
}: {
  checked: boolean;
  children: ReactNode;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      className={`min-h-10 rounded-2xl border px-3 py-2 text-left text-sm font-semibold transition ${
        checked
          ? "border-[#007ae5]/35 bg-[#f2f8ff] text-[#1c3f9a] shadow-sm shadow-[#073f73]/5"
          : "border-dashed border-[#c7c6b7] bg-white text-[#182433]/45"
      }`}
      onClick={() => onChange(!checked)}
      type="button"
    >
      {children}
    </button>
  );
}

function EyeIcon({ isVisible }: { isVisible: boolean }) {
  return (
    <span aria-hidden="true" className="relative block h-4 w-5">
      <span
        className={`absolute left-0 top-1/2 h-3 w-5 -translate-y-1/2 rounded-[50%] border ${
          isVisible ? "border-[#1c3f9a]" : "border-[#c7c6b7]"
        }`}
      />
      <span
        className={`absolute left-1/2 top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full ${
          isVisible ? "bg-[#1c3f9a]" : "bg-[#c7c6b7]"
        }`}
      />
      {!isVisible ? (
        <span className="absolute left-0 top-1/2 h-px w-5 -rotate-45 bg-[#c7c6b7]" />
      ) : null}
    </span>
  );
}

function SegmentShapePreview({ layer, segment }: { layer: OverlayLayer; segment: OverlayLayer["segments"][number] }) {
  const colors = getSegmentColors(segment, layer.key);
  const polygon = getPolygonPoints(segment);
  const bbox = segment.bbox;

  if (!polygon.length && (!bbox || bbox.length !== 4)) {
    return (
      <span
        className="block h-16 w-16 rounded-xl"
        style={{ backgroundColor: colors.fill }}
      />
    );
  }

  const points = polygon.length
    ? polygon
    : [
        [bbox?.[0] ?? 0, bbox?.[1] ?? 0],
        [bbox?.[2] ?? 0, bbox?.[1] ?? 0],
        [bbox?.[2] ?? 0, bbox?.[3] ?? 0],
        [bbox?.[0] ?? 0, bbox?.[3] ?? 0],
      ];
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const width = Math.max(1, Math.max(...xs) - minX);
  const height = Math.max(1, Math.max(...ys) - minY);
  const normalizedPoints = points
    .map(([x, y]) => `${((x - minX) / width) * 44 + 2},${((y - minY) / height) * 44 + 2}`)
    .join(" ");

  return (
    <svg
      aria-hidden="true"
      className="h-16 w-16 shrink-0"
      viewBox="0 0 48 48"
    >
      <polygon
        fill={colors.fill}
        fillOpacity="0.88"
        points={normalizedPoints}
        stroke={colors.border}
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function getLayerConfidence(layer: OverlayLayer | undefined) {
  if (!layer?.segments.length) {
    return null;
  }

  const values = layer.segments
    .map((segment) => formatConfidenceDecimal(segment))
    .filter((value): value is string => Boolean(value))
    .map(Number);

  if (!values.length) {
    return null;
  }

  const average = values.reduce((total, value) => total + value, 0) / values.length;

  return average.toFixed(2);
}

export function UploadCard({ sampleImages = [] }: UploadCardProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] =
    useState<PredictionModelKey>("full");
  const [predictions, setPredictions] = useState<PredictionResultsByModel>({});
  const [visibleLayers, setVisibleLayers] = useState(INITIAL_VISIBLE_LAYERS);
  const [hiddenSegments, setHiddenSegments] = useState<Record<string, boolean>>({});
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showBoxLabels, setShowBoxLabels] = useState(true);
  const [showBoxConfidence, setShowBoxConfidence] = useState(true);
  const [activePanelTab, setActivePanelTab] = useState<PanelTab>("setup");
  const [isDisclaimerExpanded, setIsDisclaimerExpanded] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [requestState, setRequestState] = useState<RequestState>({
    status: "idle",
    error: null,
  });

  const hasFullPrediction = Boolean(predictions.full);
  const hasBinaryPrediction = Boolean(predictions.binary);
  const hasMulticlassPrediction = Boolean(predictions.multiclass);
  const hasPredictions = Object.values(predictions).some(Boolean);
  const hasAllPartialPredictions = hasBinaryPrediction && hasMulticlassPrediction;
  const availableLayers = (["multiclass", "binary"] as const).filter((layer) =>
    predictions.full
      ? responseHasLayer(predictions.full, layer)
      : responseHasLayer(predictions[layer], layer),
  );
  const visibleOverlayLayers = getOverlayLayers(predictions, visibleLayers);
  const allOverlayLayers = getOverlayLayers(predictions, DOWNLOAD_VISIBLE_LAYERS);
  const binaryLayer = allOverlayLayers.find((layer) => layer.key === "binary");
  const multiclassLayer = allOverlayLayers.find(
    (layer) => layer.key === "multiclass",
  );
  const selectedOverlayLayers = visibleOverlayLayers
    .map((layer) => ({
      ...layer,
      segments: layer.segments.filter(
        (segment, index) => !hiddenSegments[getSegmentKey(layer.key, segment, index)],
      ),
    }))
    .filter((layer) => layer.segments.length);

  function isModelDisabled(model: PredictionModelKey) {
    if (!selectedFile || requestState.status === "loading" || hasFullPrediction) {
      return true;
    }

    if (model === "full") {
      return hasBinaryPrediction || hasMulticlassPrediction;
    }

    if (model === "binary") {
      return hasBinaryPrediction || hasAllPartialPredictions;
    }

    return hasMulticlassPrediction || hasAllPartialPredictions;
  }

  const submitDisabled =
    !selectedFile ||
    requestState.status === "loading" ||
    isModelDisabled(selectedModel);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function resetWorkspace() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    setSelectedFile(null);
    setPreviewUrl(null);
    setSelectedModel("full");
    setPredictions({});
    setHiddenSegments({});
    setVisibleLayers(INITIAL_VISIBLE_LAYERS);
    setShowBoundingBoxes(true);
    setShowBoxLabels(true);
    setShowBoxConfidence(true);
    setActivePanelTab("setup");
    setIsDisclaimerExpanded(false);
    setPan({ x: 0, y: 0 });
    setZoom(1);
    setRequestState({ status: "idle", error: null });
  }

  function loadFile(file: File | null) {
    if (!file) {
      resetWorkspace();
      return;
    }

    if (!["image/jpeg", "image/png"].includes(file.type)) {
      resetWorkspace();
      setRequestState({
        status: "error",
        error: "Please upload a JPG or PNG image.",
      });
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setSelectedModel("full");
    setPredictions({});
    setHiddenSegments({});
    setVisibleLayers(INITIAL_VISIBLE_LAYERS);
    setShowBoundingBoxes(true);
    setShowBoxLabels(true);
    setShowBoxConfidence(true);
    setActivePanelTab("setup");
    setIsDisclaimerExpanded(false);
    setPan({ x: 0, y: 0 });
    setZoom(1);
    setRequestState({ status: "idle", error: null });
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    loadFile(file);
  }

  async function loadSampleImage(sampleImage: SampleImage) {
    setRequestState({ status: "loading", error: null });

    try {
      const response = await fetch(sampleImage.src);

      if (!response.ok) {
        throw new Error("Unable to load this sample image.");
      }

      const blob = await response.blob();
      const file = new File([blob], getFilenameFromSrc(sampleImage.src), {
        type: blob.type || "image/jpeg",
      });

      loadFile(file);
    } catch (error) {
      setRequestState({
        status: "error",
        error:
          error instanceof Error
            ? error.message
            : "Unable to load this sample image.",
      });
    }
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    if (!Array.from(event.dataTransfer.types).includes("Files")) {
      return;
    }

    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }

    setIsDragging(false);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();

    if (!event.dataTransfer.files.length) {
      setIsDragging(false);
      return;
    }

    setIsDragging(false);
    loadFile(event.dataTransfer.files[0] ?? null);
  }

  function adjustZoom(amount: number) {
    setZoom((currentZoom) =>
      Math.min(2.5, Math.max(0.5, Number((currentZoom + amount).toFixed(2)))),
    );
  }

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setRequestState({
        status: "error",
        error: "Please select an image before submitting.",
      });
      return;
    }

    if (isModelDisabled(selectedModel)) {
      setRequestState({
        status: "error",
        error: "Upload a new image or choose an available model.",
      });
      return;
    }

    setRequestState({ status: "loading", error: null });

    try {
      const data = await predictScoliosisImage(selectedFile, selectedModel);
      setPredictions((currentPredictions) => ({
        ...currentPredictions,
        [selectedModel]: data,
      }));
      setVisibleLayers((currentLayers) => ({
        ...currentLayers,
        binary: selectedModel === "binary" || selectedModel === "full"
          ? true
          : currentLayers.binary,
        multiclass:
          selectedModel === "multiclass" || selectedModel === "full"
            ? true
            : currentLayers.multiclass,
      }));
      setSelectedModel((currentModel) => {
        if (currentModel === "multiclass") {
          return "binary";
        }

        if (currentModel === "binary") {
          return "multiclass";
        }

        return currentModel;
      });
      setActivePanelTab("results");
      setRequestState({ status: "idle", error: null });
    } catch (error) {
      setRequestState({
        status: "error",
        error:
          error instanceof Error
            ? error.message
            : "Unexpected prediction error.",
      });
    }
  }

  async function handleDownload(kind: DownloadKind) {
    if (!previewUrl || !selectedFile) {
      setRequestState({
        status: "error",
        error: "Select an image before downloading results.",
      });
      return;
    }

    const imageSize = getImageSize(predictions);

    if (!imageSize || !allOverlayLayers.length) {
      setRequestState({
        status: "error",
        error: "Run a prediction before downloading masks.",
      });
      return;
    }

    const layers =
      kind === "color-mask"
        ? multiclassLayer
          ? [multiclassLayer]
          : []
        : kind === "binary-selection"
          ? binaryLayer
            ? [binaryLayer]
            : []
          : selectedOverlayLayers;

    if (!layers.length) {
      setRequestState({
        status: "error",
        error: "Run the required model before downloading this image.",
      });
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = imageSize.width;
    canvas.height = imageSize.height;
    const context = canvas.getContext("2d");

    if (!context) {
      setRequestState({
        status: "error",
        error: "Unable to prepare the result download.",
      });
      return;
    }

    if (kind === "overlay") {
      const image = await loadImageElement(previewUrl);
      context.drawImage(image, 0, 0, imageSize.width, imageSize.height);
    } else {
      context.fillStyle = "#000000";
      context.fillRect(0, 0, imageSize.width, imageSize.height);
    }

    layers.forEach((layer) => {
      layer.segments.forEach((segment) => {
        if (kind === "color-mask") {
          fillSegmentPath(
            context,
            segment,
            getExportVertebraColor(getSegmentLabel(segment)),
          );
          return;
        }

        if (kind === "binary-selection") {
          fillSegmentPath(context, segment, "rgb(255, 255, 255)");
          return;
        }

        const colors = getSegmentColors(segment, layer.key);
        context.save();
        context.globalAlpha = layer.key === "binary" ? 0.28 : 0.34;
        fillSegmentPath(context, segment, colors.fill);
        context.restore();
      });

      if (kind === "overlay") {
        drawOverlayLabels(
          context,
          layer,
          showBoundingBoxes,
          showBoxLabels,
          showBoxConfidence,
          hiddenSegments,
        );
      }
    });

    const suffix =
      kind === "color-mask"
        ? "color-mask"
        : kind === "binary-selection"
          ? "binary-selection"
          : "selected-overlay";

    downloadCanvas(canvas, `${getDownloadBaseName(selectedFile)}-${suffix}.png`);
  }

  return (
    <form
      className={`grid h-full min-h-0 grid-cols-1 transition-[grid-template-columns] duration-200 ${
        isPanelOpen
          ? "lg:grid-cols-[minmax(0,1fr)_420px]"
          : "lg:grid-cols-[minmax(0,1fr)]"
      }`}
      onSubmit={handleSubmit}
    >
      <div className="relative flex min-h-0 bg-white p-3">
        <div
          className={`relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-3xl p-5 transition ${
            isDragging
              ? "bg-[#f2f8ff] ring-2 ring-[#007ae5]/45"
              : "bg-[#f2f8ff] ring-1 ring-[#c7c6b7]/40"
          }`}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <PredictionResult
            hiddenSegments={hiddenSegments}
            imageUrl={previewUrl}
            isLoading={requestState.status === "loading" && Boolean(selectedFile)}
            onPanChange={setPan}
            pan={pan}
            predictions={predictions}
            showBoxConfidence={showBoxConfidence}
            showBoxLabels={showBoxLabels}
            showBoundingBoxes={showBoundingBoxes}
            visibleLayers={visibleLayers}
            zoom={zoom}
          />

          {!isPanelOpen ? (
            <button
              aria-label="Open control panel"
              className="absolute right-4 top-4 grid h-10 w-10 place-items-center rounded-full bg-white/95 text-lg font-semibold text-[#1c3f9a] shadow-sm shadow-[#073f73]/10 transition hover:bg-[#f2f8ff]"
              onClick={() => setIsPanelOpen(true)}
              type="button"
            >
              ‹
            </button>
          ) : null}

          <div className="absolute left-4 top-1/2 grid -translate-y-1/2 gap-2 rounded-full bg-white/95 p-1 shadow-sm shadow-[#073f73]/10 ring-1 ring-[#c7c6b7]/35">
            <button
              aria-label="Zoom in"
              className="grid h-9 w-9 place-items-center rounded-full text-lg font-semibold text-[#0d1620] transition hover:bg-[#f2f8ff]"
              onClick={() => adjustZoom(0.2)}
              type="button"
            >
              +
            </button>
            <button
              aria-label="Zoom out"
              className="grid h-9 w-9 place-items-center rounded-full text-lg font-semibold text-[#0d1620] transition hover:bg-[#f2f8ff]"
              onClick={() => adjustZoom(-0.2)}
              type="button"
            >
              -
            </button>
            <button
              aria-label="Reset zoom"
              className="grid h-9 w-9 place-items-center rounded-full text-lg font-bold text-[#182433] transition hover:bg-[#f2f8ff]"
              onClick={resetView}
              type="button"
            >
              ↺
            </button>
          </div>
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center px-4 pb-4">
          <div className="pointer-events-auto flex w-full max-w-md items-center justify-center gap-2 rounded-full bg-white/95 p-2 shadow-lg shadow-[#073f73]/10 ring-1 ring-[#c7c6b7]/35">
            <input
              ref={fileInputRef}
              className="sr-only"
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
            />
            <button
              className="inline-flex h-10 items-center justify-center rounded-full bg-[#f2f8ff] px-4 text-sm font-semibold text-[#1c3f9a] transition hover:bg-[#dcedff]"
              onClick={() => fileInputRef.current?.click()}
              type="button"
            >
              Choose file
            </button>
            <Button className="h-10 flex-1" disabled={submitDisabled} type="submit">
              {requestState.status === "loading"
                ? "Segmenting..."
                : `Run ${PREDICTION_MODEL_OPTIONS.find((option) => option.key === selectedModel)?.label ?? "model"}`}
            </Button>
            {selectedFile ? (
              <button
                className="inline-flex h-10 items-center justify-center rounded-full bg-[#fff4ed] px-4 text-sm font-semibold text-[#9a3500] transition hover:bg-[#ffe6d8]"
                onClick={resetWorkspace}
                type="button"
              >
                Restart
              </button>
            ) : null}
          </div>
        </div>
      </div>

      {isPanelOpen ? (
      <aside className="min-h-0 overflow-hidden bg-white">
        <div className="flex h-full min-w-0 flex-col">
          <div className="min-w-0 border-b border-[#c7c6b7]/40 p-5">
            <div className="flex min-w-0 items-start justify-between gap-3">
              <div className="min-w-0">
                <h1 className="text-2xl font-semibold leading-tight text-[#1c3f9a]">
                  Upload X-ray
                </h1>
                <p className="mt-2 text-sm leading-6 text-[#182433]/70">
                  Select a JPG or PNG frontal spine radiograph to inspect model
                  segmentation layers.
                </p>
              </div>
              <button
                aria-label="Collapse control panel"
                className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-[#f2f8ff] text-lg font-semibold text-[#1c3f9a] transition hover:bg-[#dcedff]"
                onClick={() => setIsPanelOpen(false)}
                type="button"
              >
                ›
              </button>
            </div>
          </div>

          <div className="grid min-h-0 min-w-0 flex-1 content-start gap-5 overflow-x-hidden overflow-y-auto p-5">
            <div className="grid grid-cols-2 rounded-full bg-[#f2f8ff] p-1">
              {(["setup", "results"] as const).map((tab) => (
                <button
                  className={`h-9 rounded-full text-sm font-semibold transition ${
                    activePanelTab === tab
                      ? "bg-white text-[#1c3f9a] shadow-sm shadow-[#073f73]/10"
                      : "text-[#182433]/65 hover:text-[#1c3f9a]"
                  }`}
                  disabled={tab === "results" && !hasPredictions}
                  key={tab}
                  onClick={() => setActivePanelTab(tab)}
                  type="button"
                >
                  {tab === "setup" ? "Setup" : "Results"}
                </button>
              ))}
            </div>

            {activePanelTab === "setup" ? (
              <>
                {sampleImages.length ? (
                  <fieldset className="grid min-w-0 gap-2">
                    <legend className="text-sm font-semibold text-[#0d1620]">
                      Examples
                    </legend>
                    <div className="grid min-w-0 grid-cols-3 gap-2">
                      {sampleImages.map((sampleImage) => (
                        <button
                          className="group min-w-0 overflow-hidden rounded-2xl bg-[#f2f8ff] text-left ring-1 ring-[#c7c6b7]/35 transition hover:bg-[#dcedff] hover:ring-[#007ae5]/35 focus:outline-none focus:ring-2 focus:ring-[#007ae5]/35"
                          key={sampleImage.src}
                          onClick={() => void loadSampleImage(sampleImage)}
                          type="button"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            alt={sampleImage.alt}
                            className="h-24 w-full object-cover grayscale transition group-hover:grayscale-0"
                            src={sampleImage.src}
                          />
                          <span className="block truncate px-2 py-2 text-xs font-semibold text-[#1c3f9a]">
                            {sampleImage.label}
                          </span>
                        </button>
                      ))}
                    </div>
                  </fieldset>
                ) : null}

                <fieldset className="grid min-w-0 gap-2">
                  <legend className="text-sm font-semibold text-[#0d1620]">
                    Models
                  </legend>
                  <div className="grid min-w-0 gap-2">
                    {PREDICTION_MODEL_OPTIONS.map((option) => (
                      <label
                        className={`min-w-0 rounded-2xl p-3 transition ${
                          isModelDisabled(option.key)
                            ? "cursor-not-allowed bg-[#fbfaf0] opacity-55"
                            : selectedModel === option.key
                              ? "bg-[#f2f8ff] ring-2 ring-[#007ae5]/35"
                              : "cursor-pointer bg-white ring-1 ring-[#c7c6b7]/35 hover:bg-[#f2f8ff]"
                        }`}
                        key={option.key}
                      >
                        <input
                          checked={selectedModel === option.key}
                          className="sr-only"
                          disabled={isModelDisabled(option.key)}
                          name="prediction-model"
                          onChange={() => setSelectedModel(option.key)}
                          suppressHydrationWarning
                          type="radio"
                          value={option.key}
                        />
                        <span className="block min-w-0 break-words text-sm font-semibold text-[#0d1620]">
                          {option.label}
                        </span>
                        <span className="mt-1 block min-w-0 break-words text-xs leading-5 text-[#182433]/70">
                          {option.description}
                        </span>
                      </label>
                    ))}
                  </div>
                </fieldset>
              </>
            ) : (
              <>
                {availableLayers.length ? (
                  <fieldset className="grid min-w-0 gap-2">
                    <legend className="text-sm font-semibold text-[#0d1620]">
                      Layers
                    </legend>
                    <div className="grid min-w-0 gap-2">
                      {availableLayers.map((layer) => (
                        <ToggleButton
                          checked={visibleLayers[layer]}
                          key={layer}
                          onChange={(checked) =>
                            setVisibleLayers((currentLayers) => ({
                              ...currentLayers,
                              [layer]: checked,
                            }))
                          }
                        >
                          <span className="flex min-w-0 items-center justify-between gap-3">
                            <span className="min-w-0 break-words">
                              {getLayerLabel(layer)}
                            </span>
                            <span>{visibleLayers[layer] ? "On" : "Off"}</span>
                          </span>
                        </ToggleButton>
                      ))}
                      {availableLayers.includes("multiclass") ? (
                        <div className="grid grid-cols-3 gap-2">
                          <ToggleButton
                            checked={showBoundingBoxes}
                            onChange={setShowBoundingBoxes}
                          >
                            Boxes
                          </ToggleButton>
                          <ToggleButton
                            checked={showBoxLabels}
                            onChange={setShowBoxLabels}
                          >
                            Labels
                          </ToggleButton>
                          <ToggleButton
                            checked={showBoxConfidence}
                            onChange={setShowBoxConfidence}
                          >
                            Scores
                          </ToggleButton>
                        </div>
                      ) : null}
                    </div>
                  </fieldset>
                ) : null}

                <section className="grid min-w-0 gap-3">
                  <h2 className="text-sm font-semibold text-[#0d1620]">
                    Prediction details
                  </h2>
                  {multiclassLayer ? (
                    <div className="grid min-w-0 gap-2">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-[#1c3f9a]">
                          Vertebrae
                        </p>
                        <span className="rounded-full bg-[#f2f8ff] px-2 py-1 text-xs font-semibold text-[#1c3f9a]">
                          {multiclassLayer.segments.length}
                        </span>
                      </div>
                      <div className="grid max-h-96 min-w-0 gap-3 overflow-y-auto pr-1">
                        {multiclassLayer.segments.map((segment, index) => {
                          const segmentKey = getSegmentKey(
                            multiclassLayer.key,
                            segment,
                            index,
                          );
                          const isVisible = !hiddenSegments[segmentKey];
                          const confidence = formatConfidenceDecimal(segment);

                          return (
                            <div
                              className={`grid min-w-0 grid-cols-[auto_1fr_auto] items-center gap-4 rounded-2xl border px-4 py-3 transition ${
                                isVisible
                                  ? "border-[#c7c6b7]/45 bg-white"
                                  : "border-dashed border-[#c7c6b7] bg-white opacity-50"
                              }`}
                              key={segmentKey}
                            >
                              <SegmentShapePreview
                                layer={multiclassLayer}
                                segment={segment}
                              />
                              <div className="min-w-0">
                                <p className="truncate text-sm font-semibold text-[#0d1620]">
                                  {getSegmentLabel(segment)}
                                </p>
                                <p className="mt-1 text-xs font-semibold text-[#182433]/60">
                                  Score {confidence ?? "n/a"}
                                </p>
                              </div>
                              <button
                                aria-label={`${isVisible ? "Hide" : "Show"} ${getSegmentLabel(segment)}`}
                                className={`grid h-9 w-9 place-items-center rounded-full transition ${
                                  isVisible
                                    ? "bg-[#f2f8ff] text-[#1c3f9a]"
                                    : "bg-white text-[#182433]/45"
                                }`}
                                onClick={() =>
                                  setHiddenSegments((currentSegments) => ({
                                    ...currentSegments,
                                    [segmentKey]: isVisible,
                                  }))
                                }
                                type="button"
                              >
                                <EyeIcon isVisible={isVisible} />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : null}

                  {binaryLayer ? (
                    <div className="grid min-w-0 gap-2 rounded-2xl bg-white p-3 ring-1 ring-[#c7c6b7]/35">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-[#1c3f9a]">
                          Binary mask
                        </p>
                        <span className="rounded-full bg-[#f2f8ff] px-2 py-1 text-xs font-semibold text-[#1c3f9a]">
                          {binaryLayer.segments.length ? "Detected" : "Empty"}
                        </span>
                      </div>
                      <div className="grid min-w-0 grid-cols-[auto_1fr_auto] items-center gap-2 rounded-xl bg-[#fbfaf0] px-3 py-2">
                        <span className="h-3 w-3 rounded-full bg-[#0078E5]" />
                        <span className="min-w-0 truncate text-sm font-semibold text-[#0d1620]">
                          Overall selection
                        </span>
                        <span className="text-xs font-semibold text-[#182433]/65">
                          {getLayerConfidence(binaryLayer) ?? "n/a"}
                        </span>
                      </div>
                    </div>
                  ) : null}

                  {!allOverlayLayers.length ? (
                    <p className="rounded-2xl bg-[#fbfaf0] px-4 py-3 text-sm leading-6 text-[#182433]/70">
                      Run a model to inspect prediction layers.
                    </p>
                  ) : null}
                </section>

                <section className="grid min-w-0 gap-2">
                  <h2 className="text-sm font-semibold text-[#0d1620]">
                    Downloads
                  </h2>
                  <div className="grid gap-2">
                    <button
                      className="h-10 rounded-full bg-[#1c3f9a] px-4 text-sm font-semibold text-white transition hover:bg-[#007ae5] disabled:cursor-not-allowed disabled:bg-[#c7c6b7]"
                      disabled={!multiclassLayer}
                      onClick={() => void handleDownload("color-mask")}
                      type="button"
                    >
                      Download color mask
                    </button>
                    <button
                      className="h-10 rounded-full bg-[#1c3f9a] px-4 text-sm font-semibold text-white transition hover:bg-[#007ae5] disabled:cursor-not-allowed disabled:bg-[#c7c6b7]"
                      disabled={!binaryLayer}
                      onClick={() => void handleDownload("binary-selection")}
                      type="button"
                    >
                      Download binary selection
                    </button>
                    <button
                      className="h-10 rounded-full bg-[#1c3f9a] px-4 text-sm font-semibold text-white transition hover:bg-[#007ae5] disabled:cursor-not-allowed disabled:bg-[#c7c6b7]"
                      disabled={!selectedOverlayLayers.length}
                      onClick={() => void handleDownload("overlay")}
                      type="button"
                    >
                      Download selected overlay
                    </button>
                  </div>
                </section>
              </>
            )}

            <div className="min-w-0 rounded-2xl bg-[#fbfaf0] p-3 ring-1 ring-[#c7c6b7]/35">
              <p className="text-xs font-semibold uppercase tracking-wide text-[#182433]/55">
                Current file
              </p>
              <p className="mt-1 break-words text-sm font-semibold text-[#0d1620]">
                {selectedFile?.name ?? "No image selected"}
              </p>
            </div>

            {requestState.status === "error" ? (
              <p className="rounded-2xl bg-[#fff0ed] px-4 py-3 text-sm leading-6 text-[#9a2600]">
                {requestState.error}
              </p>
            ) : null}
          </div>

          <button
            className="min-w-0 border-t border-[#c7c6b7]/40 bg-[#f2f8ff] px-5 py-3 text-left text-xs leading-5 text-[#182433]/70 transition hover:bg-[#dcedff]"
            onClick={() =>
              setIsDisclaimerExpanded((currentValue) => !currentValue)
            }
            type="button"
          >
            <span className="flex items-center justify-between gap-3">
              <span className="font-medium text-[#0d1620]">
                Research demo
              </span>
              <span className="text-sm font-semibold text-[#1c3f9a]">
                {isDisclaimerExpanded ? "−" : "+"}
              </span>
            </span>
            {isDisclaimerExpanded ? (
              <>
                <span className="mt-1 block">
                  AI-powered scoliosis detection workspace.
                </span>
                <span className="mt-1 block">{PROJECT_DISCLAIMER}</span>
              </>
            ) : null}
          </button>
        </div>
      </aside>
      ) : null}
    </form>
  );
}
