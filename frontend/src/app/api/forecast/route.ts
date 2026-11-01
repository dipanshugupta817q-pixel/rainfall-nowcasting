import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

type RawFrame = {
  horizon_minutes: number;
  grid: number[][];
  min: number;
  max: number;
  mean: number;
  valid_time?: string;
};

type RawForecast = {
  generated_at?: string;
  valid_from?: string;
  grid: {
    height: number;
    width: number;
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  frames: RawFrame[];
};

/**
 * Stable frontend boundary.
 *
 * Replace only the adapter inside this route when the final forecasting
 * service changes. The browser never receives model/source metadata.
 */
export async function GET() {
  try {
    // Temporary adapter for the existing forecast artifact.
    // Later this can call the fused forecasting service without changing
    // any frontend component.
    const filePath = path.join(
      process.cwd(),
      "public",
      "data",
      "model_a_forecast.json",
    );

    const raw = JSON.parse(
      await fs.readFile(filePath, "utf8"),
    ) as RawForecast & Record<string, unknown>;

    if (!raw.grid || !Array.isArray(raw.frames) || raw.frames.length === 0) {
      return NextResponse.json(
        { error: "Forecast source contains no usable frames." },
        { status: 502 },
      );
    }

    // Intentionally construct a new object so model/architecture/source
    // metadata cannot accidentally leak into the browser contract.
    const response = {
      generated_at: raw.generated_at,
      valid_from: raw.valid_from,
      unit: "mm/hr" as const,
      grid: raw.grid,
      frames: raw.frames,
    };

    return NextResponse.json(response, {
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error("Forecast adapter error:", error);

    return NextResponse.json(
      { error: "Forecast service is currently unavailable." },
      { status: 503 },
    );
  }
}
