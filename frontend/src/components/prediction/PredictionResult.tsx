"use client";

import { useState } from "react";
import type { PointerEvent } from "react";
import { GENERAL_SPINE_COLORS, getVertebraColors } from "@/lib/constants";
import type {
  PredictionResponse,
  PredictionResultsByModel,
  PredictionSegment,
} from "@/types/prediction";

type LayerKey = "binary" | "multiclass";

type PredictionResultProps = {
  imageUrl: string | null;
  predictions?: PredictionResultsByModel;
  showBoundingBoxes: boolean;
  visibleLayers: Record<LayerKey, boolean>;
  zoom: number;
  onPanChange: (pan: { x: number; y: number }) => void;
  pan: { x: number; y: number };
};

type OverlayLayer = {
  key: LayerKey;
  label: string;
  response: PredictionResponse;
  segments: PredictionSegment[];
};

function getLayerLabel(key: LayerKey) {
  return key === "binary" ? "Binary mask" : "Vertebra labels";
}

function getSegmentLabel(segment: PredictionSegment) {
  return segment.label ?? `Class ${segment.class_id ?? segment.id ?? "?"}`;
}

function getSegmentColors(segment: PredictionSegment, layerKey: LayerKey) {
  if (layerKey === "binary") {
    return GENERAL_SPINE_COLORS;
  }

  return getVertebraColors(getSegmentLabel(segment));
}

function getSegmentsForLayer(
  response: PredictionResponse,
  layerKey: LayerKey,
): PredictionSegment[] {
  const modelResult = response.results?.[layerKey];

  if (Array.isArray(modelResult?.segments)) {
    return modelResult.segments;
  }

  if (layerKey === "binary" && response.results) {
    const binaryLike = Object.values(response.results).find(
      (result) => result.type === "binary" && Array.isArray(result.segments),
    );

    return binaryLike?.segments ?? [];
  }

  if (layerKey === "multiclass" && response.results) {
    const multiclassLike = Object.values(response.results).find(
      (result) =>
        result.type === "multiclass" && Array.isArray(result.segments),
    );

    return multiclassLike?.segments ?? [];
  }

  return [];
}

function getOverlayLayers(
  predictions: PredictionResultsByModel = {},
  visibleLayers: Record<LayerKey, boolean>,
): OverlayLayer[] {
  const fullResponse = predictions.full;
  const sources: { key: LayerKey; response?: PredictionResponse }[] = fullResponse
    ? [
        { key: "multiclass", response: fullResponse },
        { key: "binary", response: fullResponse },
      ]
    : [
        { key: "multiclass", response: predictions.multiclass },
        { key: "binary", response: predictions.binary },
      ];

  return sources.flatMap(({ key, response }) => {
    if (!response || !visibleLayers[key]) {
      return [];
    }

    const segments = getSegmentsForLayer(response, key);

    if (!segments.length) {
      return [];
    }

    return [
      {
        key,
        label: getLayerLabel(key),
        response,
        segments,
      },
    ];
  });
}

function getImageSize(predictions: PredictionResultsByModel = {}) {
  const response = predictions.full ?? predictions.multiclass ?? predictions.binary;

  if (!response?.image_width || !response.image_height) {
    return null;
  }

  return {
    height: response.image_height,
    width: response.image_width,
  };
}

export function PredictionResult({
  imageUrl,
  onPanChange,
  pan,
  predictions = {},
  showBoundingBoxes,
  visibleLayers,
  zoom,
}: PredictionResultProps) {
  const [dragStart, setDragStart] = useState<{
    panX: number;
    panY: number;
    pointerId: number;
    x: number;
    y: number;
  } | null>(null);
  const imageSize = getImageSize(predictions);
  const overlayLayers = getOverlayLayers(predictions, visibleLayers);

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (zoom <= 1) {
      return;
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    setDragStart({
      panX: pan.x,
      panY: pan.y,
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!dragStart) {
      return;
    }

    onPanChange({
      x: dragStart.panX + event.clientX - dragStart.x,
      y: dragStart.panY + event.clientY - dragStart.y,
    });
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    if (dragStart?.pointerId === event.pointerId) {
      setDragStart(null);
    }
  }

  if (!imageUrl) {
    return (
      <div className="grid place-items-center text-center">
        <div className="grid h-16 w-16 place-items-center rounded-full border-2 border-dashed border-[#007ae5]/30 bg-white text-3xl text-[#007ae5] shadow-sm shadow-[#073f73]/10">
          +
        </div>
        <p className="mt-4 max-w-xs text-sm font-semibold text-[#0d1620]">
          Drop a frontal spine radiograph here
        </p>
        <p className="mt-1 max-w-xs text-sm leading-6 text-[#182433]/65">
          JPG and PNG images are supported.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full items-center justify-center overflow-auto p-4">
      <div
        className={`relative inline-block max-w-full origin-center select-none ${
          zoom > 1 ? "cursor-grab active:cursor-grabbing" : ""
        }`}
        onPointerCancel={handlePointerUp}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transition: dragStart ? "none" : "transform 160ms ease",
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          alt="Selected X-ray preview"
          className="max-h-[calc(100vh-12rem)] w-auto max-w-full rounded-2xl object-contain shadow-sm shadow-[#073f73]/10"
          draggable={false}
          src={imageUrl}
        />
        {imageSize && overlayLayers.length ? (
          <svg
            aria-label="Prediction geometry overlay"
            className="pointer-events-none absolute inset-0 h-full w-full"
            preserveAspectRatio="none"
            viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          >
            {overlayLayers.map((layer) =>
              layer.segments.map((segment, index) => {
                const colors = getSegmentColors(segment, layer.key);
                const polygonPoints = segment.polygon
                  ?.map((point) => point.join(","))
                  .join(" ");
                const bbox = segment.bbox;
                const showBox = layer.key === "multiclass" && showBoundingBoxes;

                return (
                  <g key={`${layer.key}-${getSegmentLabel(segment)}-${index}`}>
                    {polygonPoints ? (
                      <polygon
                        fill={colors.fill}
                        fillOpacity={layer.key === "binary" ? "0.28" : "0.3"}
                        points={polygonPoints}
                        stroke={colors.border}
                        strokeLinejoin="round"
                        strokeWidth={layer.key === "binary" ? "2" : "2.5"}
                      />
                    ) : null}
                    {showBox && bbox && bbox.length === 4 ? (
                      <rect
                        fill="none"
                        height={Math.max(0, bbox[3] - bbox[1])}
                        stroke={colors.border}
                        strokeDasharray="8 6"
                        strokeWidth="2.5"
                        width={Math.max(0, bbox[2] - bbox[0])}
                        x={bbox[0]}
                        y={bbox[1]}
                      />
                    ) : null}
                    {showBox && bbox && bbox.length === 4 ? (
                      <text
                        fill="white"
                        fontSize="22"
                        fontWeight="700"
                        paintOrder="stroke"
                        stroke="#0d1620"
                        strokeWidth="5"
                        x={bbox[0]}
                        y={Math.max(24, bbox[1] - 8)}
                      >
                        {getSegmentLabel(segment)}
                      </text>
                    ) : null}
                  </g>
                );
              }),
            )}
          </svg>
        ) : null}
      </div>
    </div>
  );
}
