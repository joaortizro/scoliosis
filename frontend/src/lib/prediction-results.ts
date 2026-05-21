import { GENERAL_SPINE_COLORS, getVertebraColors } from "@/lib/constants";
import type {
  PredictionResponse,
  PredictionResultsByModel,
  PredictionSegment,
} from "@/types/prediction";

export type LayerKey = "binary" | "multiclass";

export type OverlayLayer = {
  key: LayerKey;
  label: string;
  response: PredictionResponse;
  segments: PredictionSegment[];
};

export function getLayerLabel(key: LayerKey) {
  return key === "binary" ? "Binary mask" : "Vertebra labels";
}

export function getSegmentLabel(segment: PredictionSegment) {
  return segment.label ?? `Class ${segment.class_id ?? segment.id ?? "?"}`;
}

export function getSegmentConfidence(segment: PredictionSegment) {
  const value =
    typeof segment.confidence === "number"
      ? segment.confidence
      : typeof segment.score === "number"
        ? segment.score
        : typeof segment.probability === "number"
          ? segment.probability
          : null;

  if (value === null || Number.isNaN(value)) {
    return null;
  }

  return value <= 1 ? value * 100 : value;
}

export function formatConfidence(segment: PredictionSegment) {
  const value = getSegmentConfidence(segment);

  return value === null ? null : `${Math.round(value)}%`;
}

export function formatConfidenceDecimal(segment: PredictionSegment) {
  const value = getSegmentConfidence(segment);

  return value === null ? null : (value / 100).toFixed(2);
}

export function getSegmentKey(
  layerKey: LayerKey,
  segment: PredictionSegment,
  index: number,
) {
  return `${layerKey}-${getSegmentLabel(segment)}-${segment.class_id ?? segment.id ?? index}-${index}`;
}

export function getSegmentClassId(segment: PredictionSegment) {
  if (typeof segment.class_id === "number") {
    return segment.class_id;
  }

  if (typeof segment.id === "number") {
    return segment.id;
  }

  const label = segment.label?.trim().toUpperCase();
  const thoracicMatch = label?.match(/^T(\d{1,2})$/);
  const lumbarMatch = label?.match(/^L([1-5])$/);

  if (thoracicMatch) {
    return Number(thoracicMatch[1]);
  }

  if (lumbarMatch) {
    return 12 + Number(lumbarMatch[1]);
  }

  return 1;
}

export function getSegmentColors(segment: PredictionSegment, layerKey: LayerKey) {
  if (layerKey === "binary") {
    return GENERAL_SPINE_COLORS;
  }

  return getVertebraColors(getSegmentLabel(segment));
}

export function getSegmentsForLayer(
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

export function getOverlayLayers(
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

export function getImageSize(predictions: PredictionResultsByModel = {}) {
  const response = predictions.full ?? predictions.multiclass ?? predictions.binary;

  if (!response?.image_width || !response.image_height) {
    return null;
  }

  return {
    height: response.image_height,
    width: response.image_width,
  };
}

export function getPolygonPoints(segment: PredictionSegment) {
  if (!Array.isArray(segment.polygon)) {
    return [];
  }

  return segment.polygon.filter(
    (point): point is [number, number] =>
      Array.isArray(point) &&
      point.length >= 2 &&
      typeof point[0] === "number" &&
      typeof point[1] === "number",
  );
}
