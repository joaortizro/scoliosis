export type VertebraPrediction = {
  label?: string;
  confidence?: number;
  centroid_x?: number;
  centroid_y?: number;
  area_px?: number;
  source?: string;
  [key: string]: unknown;
};

export type PredictionResponse = {
  data?: {
    vertebrae?: VertebraPrediction[];
    image_base64?: string;
    [key: string]: unknown;
  };
  vertebrae?: VertebraPrediction[];
  image_base64?: string;
  prediction?: string;
  label?: string;
  confidence?: number;
  probability?: number;
  cobb_angle?: number;
  message?: string;
  [key: string]: unknown;
};

export type PredictionState =
  | { status: "idle"; data: null; error: null }
  | { status: "loading"; data: null; error: null }
  | { status: "success"; data: PredictionResponse; error: null }
  | { status: "error"; data: null; error: string };
