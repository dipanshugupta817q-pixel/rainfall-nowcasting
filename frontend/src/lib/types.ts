export type ForecastFrame = {
  horizon_minutes: number;
  grid: number[][];
  min: number;
  max: number;
  mean: number;
};

export type ForecastGrid = {
  height: number;
  width: number;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
};

export type ForecastLocation = {
  lat: number;
  lon: number;
  label?: string;
};

export type ForecastWarning = {
  type: "heavy_rain" | "flood" | "inundation" | "other";
  severity: "low" | "moderate" | "high" | "extreme";
  message: string;
  validFrom?: string;
  validUntil?: string;
};

export type ForecastData = {
  generatedAt?: string;
  validUntil?: string;

  location?: ForecastLocation;

  unit: string;

  grid: ForecastGrid;

  frames: ForecastFrame[];

  warnings?: ForecastWarning[];

  inundation?: {
    riskLevel: string;
    affectedArea?: number;
    grid?: number[][];
  };
};
