"use client";

import { useEffect, useRef, useState } from "react";

type GeoapifyFeature = {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
  properties?: {
    place_id?: string;
    name?: string;
    formatted?: string;
    address_line1?: string;
    address_line2?: string;
    city?: string;
    town?: string;
    village?: string;
    suburb?: string;
    district?: string;
    state?: string;
    country?: string;
    country_code?: string;
    result_type?: string;
    category?: string;
    lat?: number;
    lon?: number;
    [key: string]: unknown;
  };
};

type GeoapifyResponse = {
  features?: GeoapifyFeature[];
};

export type LocationTarget = {
  longitude: number;
  latitude: number;
  label: string;
  description?: string;
};

export type ForecastDomain = {
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
};

type LocationSearchProps = {
  onLocationSelect: (target: LocationTarget) => void;
  onOutsideDomain?: (target: LocationTarget) => void;
  domain?: ForecastDomain;
  resolutionKm?: number;
};

const GEOAPIFY_KEY = process.env.NEXT_PUBLIC_GEOAPIFY_KEY;

const DEFAULT_DOMAIN: ForecastDomain = {
  latMin: 28.05,
  latMax: 29.95,
  lonMin: 76.05,
  lonMax: 78.95,
};

const MEERUT_BIAS = {
  longitude: 77.7064,
  latitude: 28.9845,
};

function isInsideDomain(
  longitude: number,
  latitude: number,
  domain: ForecastDomain
) {
  return (
    longitude >= domain.lonMin &&
    longitude <= domain.lonMax &&
    latitude >= domain.latMin &&
    latitude <= domain.latMax
  );
}

function getFeatureLabel(feature: GeoapifyFeature) {
  const p = feature.properties;

  return (
    p?.name ||
    p?.address_line1 ||
    p?.formatted?.split(",")[0] ||
    p?.village ||
    p?.town ||
    p?.city ||
    "Selected location"
  );
}

function getFeatureDescription(feature: GeoapifyFeature) {
  const p = feature.properties;

  return (
    p?.formatted ||
    [p?.address_line1, p?.address_line2]
      .filter(Boolean)
      .join(", ") ||
    ""
  );
}

function scoreFeature(
  feature: GeoapifyFeature,
  query: string
) {
  const p = feature.properties;

  const q = query.toLowerCase().trim();

  const name = String(p?.name ?? "").toLowerCase();
  const formatted = String(p?.formatted ?? "").toLowerCase();
  const address = String(p?.address_line1 ?? "").toLowerCase();
  const address2 = String(p?.address_line2 ?? "").toLowerCase();
  const resultType = String(p?.result_type ?? "").toLowerCase();
  const category = String(p?.category ?? "").toLowerCase();

  let score = 0;

  /*
   * Exact geographic-name matches must dominate the ranking.
   */
  if (name === q) score += 1000;
  else if (name.startsWith(q)) score += 500;
  else if (name.includes(q)) score += 250;

  /*
   * A query matching the formatted location is useful,
   * but should remain below a direct name match.
   */
  if (formatted === q) score += 400;
  if (formatted.startsWith(`${q},`)) score += 300;
  if (formatted.includes(q)) score += 100;

  if (address === q) score += 200;
  if (address.startsWith(q)) score += 100;
  if (address.includes(q)) score += 50;
  if (address2.includes(q)) score += 25;

  /*
   * Geographic / administrative results are preferred
   * over individual POIs and businesses.
   */
  if (
    resultType.includes("city") ||
    resultType.includes("town") ||
    resultType.includes("district") ||
    resultType.includes("state") ||
    resultType.includes("county") ||
    resultType.includes("locality") ||
    resultType.includes("municipality") ||
    resultType.includes("village") ||
    resultType.includes("suburb") ||
    resultType.includes("neighbourhood") ||
    resultType.includes("neighborhood")
  ) {
    score += 300;
  }

  if (
    category.includes("place") ||
    category.includes("administrative")
  ) {
    score += 150;
  }

  /*
   * POIs should not outrank a geographic location merely
   * because they contain the search text.
   */
  if (
    resultType.includes("amenity") ||
    resultType.includes("poi") ||
    resultType.includes("building")
  ) {
    score -= 100;
  }

  /*
   * Education can still be useful when explicitly searched,
   * but it should not receive a blanket ranking boost.
   */
  if (
    category.includes("education") ||
    category.includes("university") ||
    category.includes("college")
  ) {
    score += q.includes("university") ||
      q.includes("college") ||
      q.includes("school")
      ? 50
      : 0;
  }

  return score;
}

function buildGeoapifyUrl(text: string) {
  const params = new URLSearchParams({
    text,
    apiKey: GEOAPIFY_KEY ?? "",
    limit: "10",
    lang: "en",
    filter: "countrycode:in",
    bias: `proximity:${MEERUT_BIAS.longitude},${MEERUT_BIAS.latitude}`,
  });

  return `https://api.geoapify.com/v1/geocode/autocomplete?${params.toString()}`;
}

export default function LocationSearch({
  onLocationSelect,
  onOutsideDomain,
  domain = DEFAULT_DOMAIN,
}: LocationSearchProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<GeoapifyFeature[]>([]);
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestRef = useRef(0);

  function scheduleCollapse() {
    if (collapseTimerRef.current) {
      clearTimeout(collapseTimerRef.current);
    }

    collapseTimerRef.current = setTimeout(() => {
      setExpanded(false);
      setOpen(false);
    }, 3000);
  }

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }

      if (collapseTimerRef.current) {
        clearTimeout(collapseTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
    };
  }, []);

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    const trimmed = query.trim();

    if (trimmed.length < 2) {
      setResults([]);
      setOpen(false);
      setLoading(false);
      setSearchError(null);
      return;
    }

    if (!GEOAPIFY_KEY) {
      setSearchError("Location search is temporarily unavailable.");
      setOpen(true);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      const requestId = ++requestRef.current;

      setLoading(true);
      setSearchError(null);
      setOpen(true);

      try {
        const response = await fetch(buildGeoapifyUrl(trimmed));

        if (!response.ok) {
          throw new Error(
            `Geoapify request failed: ${response.status}`
          );
        }

        const data: GeoapifyResponse = await response.json();

        if (requestId !== requestRef.current) {
          return;
        }

        let features = data.features ?? [];

        /*
         * If Geoapify returns weak/no results for a short locality
         * query, retry with explicit Meerut context.
         */
        if (features.length === 0) {
          const contextualQuery = `${trimmed}, Meerut, Uttar Pradesh, India`;

          const fallbackResponse = await fetch(
            buildGeoapifyUrl(contextualQuery)
          );

          if (fallbackResponse.ok) {
            const fallbackData: GeoapifyResponse =
              await fallbackResponse.json();

            features = fallbackData.features ?? [];
          }
        }

        features = features
          .sort(
            (a, b) =>
              scoreFeature(b, trimmed) -
              scoreFeature(a, trimmed)
          )
          .slice(0, 8);

        setResults(features);
      } catch (error) {
        console.error("Location search error:", error);

        if (requestId === requestRef.current) {
          setResults([]);
          setSearchError("Unable to search this location.");
        }
      } finally {
        if (requestId === requestRef.current) {
          setLoading(false);
        }
      }
    }, 350);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [query]);

  function handleSelect(feature: GeoapifyFeature) {
    const coordinates = feature.geometry.coordinates;

    const longitude =
      Number(feature.properties?.lon ?? coordinates[0]);

    const latitude =
      Number(feature.properties?.lat ?? coordinates[1]);

    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) {
      setSearchError("This result does not have a valid map position.");
      return;
    }

    const target: LocationTarget = {
      longitude,
      latitude,
      label: getFeatureLabel(feature),
      description: getFeatureDescription(feature),
    };

    setQuery(target.label);
    setResults([]);
    setOpen(false);
    setSearchError(null);

    console.log("[LocationSearch] Selected location:", {
      label: target.label,
      latitude,
      longitude,
    });

    console.log("[LocationSearch] Active forecast domain:", domain);

    console.log(
      "[LocationSearch] Inside domain:",
      isInsideDomain(longitude, latitude, domain)
    );

    if (isInsideDomain(longitude, latitude, domain)) {
      setNotice(null);
      onLocationSelect(target);
      scheduleCollapse();
    } else {
      setNotice(
        "Forecast coverage is currently limited to the Meerut region."
      );
      setOpen(true);
      onOutsideDomain?.(target);
    }
  }

  function clearSearch() {
    if (collapseTimerRef.current) {
      clearTimeout(collapseTimerRef.current);
    }

    setQuery("");
    setResults([]);
    setOpen(false);
    setNotice(null);
    setSearchError(null);
  }

  function openSearch() {
    if (collapseTimerRef.current) {
      clearTimeout(collapseTimerRef.current);
    }

    setExpanded(true);
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLInputElement>
  ) {
    if (event.key === "Escape") {
      setOpen(false);
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();

      if (results.length > 0) {
        handleSelect(results[0]);
      }
    }
  }

  return (
    <div
      ref={wrapperRef}
      className="absolute left-2 top-1 z-40"
    >
      {!expanded ? (
        <button
          type="button"
          onClick={openSearch}
          className="flex h-11 w-11 items-center justify-center rounded-xl border border-white/10 bg-black/45 text-slate-300 shadow-2xl backdrop-blur-md transition hover:bg-black/60 hover:text-white"
          aria-label="Search location"
          title="Search location"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.5-3.5" />
          </svg>
        </button>
      ) : (
        <div className="w-[min(430px,calc(100vw-1.5rem))]">
          <div className="relative">
            <div className="flex items-center rounded-xl border border-white/10 bg-black/45 shadow-2xl backdrop-blur-md">
              <div className="pl-3 text-slate-400">
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  aria-hidden="true"
                >
                  <circle cx="11" cy="11" r="7" />
                  <path d="m20 20-3.5-3.5" />
                </svg>
              </div>

              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onFocus={() => {
                  if (query.trim().length >= 2) {
                    setOpen(true);
                  }
                }}
                onKeyDown={handleKeyDown}
                placeholder="Search village, locality, road, university..."
                className="h-11 min-w-0 flex-1 bg-transparent px-3 text-sm text-white outline-none placeholder:text-slate-500"
                aria-label="Search location"
              />

              {loading && (
                <div className="mr-3 h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-cyan-400" />
              )}

              {!loading && query && (
                <button
                  type="button"
                  onClick={clearSearch}
                  className="mr-2 rounded-md p-1.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
                  aria-label="Clear location search"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                  >
                    <path d="M6 6l12 12M18 6 6 18" />
                  </svg>
                </button>
              )}
            </div>

            {open && (
              <div className="mt-2 overflow-hidden rounded-xl border border-white/10 bg-[#111918]/90 shadow-2xl backdrop-blur-xl">
                {loading && (
                  <div className="px-4 py-3 text-xs text-slate-500">
                    Searching locations...
                  </div>
                )}

                {!loading && searchError && (
                  <div className="px-4 py-3 text-xs text-red-300">
                    {searchError}
                  </div>
                )}

                {!loading &&
                  !searchError &&
                  results.length === 0 &&
                  query.trim().length >= 2 && (
                    <div className="px-4 py-3 text-xs text-slate-500">
                      No matching locations found.
                    </div>
                  )}

                {!loading && results.length > 0 && (
                  <div className="max-h-72 overflow-y-auto py-1">
                    {results.map((feature, index) => {
                      const label = getFeatureLabel(feature);
                      const description =
                        getFeatureDescription(feature);

                      return (
                        <button
                          key={
                            feature.properties?.place_id ??
                            `${label}-${index}`
                          }
                          type="button"
                          onClick={() => handleSelect(feature)}
                          className="flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-white/5"
                        >
                          <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-cyan-300">
                            <svg
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="1.8"
                              aria-hidden="true"
                            >
                              <path d="M12 21s7-6.1 7-12a7 7 0 1 0-14 0c0 5.9 7 12 7 12Z" />
                              <circle cx="12" cy="9" r="2.2" />
                            </svg>
                          </span>

                          <span className="min-w-0">
                            <span className="block truncate text-sm font-medium text-slate-200">
                              {label}
                            </span>

                            {description &&
                              description !== label && (
                                <span className="mt-0.5 block truncate text-[10px] text-slate-500">
                                  {description}
                                </span>
                              )}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="border-t border-white/5 px-4 py-2 text-[9px] text-slate-600">
                  Location data provided by Geoapify
                </div>
              </div>
            )}

            {notice && (
              <div className="mt-2 rounded-xl border border-amber-400/10 bg-[#111918]/90 px-4 py-3 shadow-xl backdrop-blur-xl">
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400" />

                  <div>
                    <div className="text-[11px] font-medium text-slate-300">
                      Forecast coverage unavailable
                    </div>

                    <div className="mt-1 text-[10px] leading-relaxed text-slate-500">
                      This location is outside the current forecast
                      domain. Expanded regional coverage is currently
                      being integrated.
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}






