import { FACETS, FAMILIES } from "./model/families";
import {
  applyFacetFilter,
  applyNameFilter,
  applyTextFilter,
  familyHasTextVariant,
} from "./model/rowFilters";
import type {
  FacetableRow,
  NamedRow,
  TextVariantRow,
} from "./model/rowFilters";
import {
  genreByGenreColumns,
  genreOverallColumns,
  parallelismByTypeColumns,
  parallelismOverallColumns,
  trajectoryByGenreColumns,
  trajectoryOverallColumns,
} from "./model/tableColumns";
import type {
  GenreByGenreRow,
  GenreOverallRow,
  ParallelismByTypeRow,
  ParallelismOverallRow,
  TrajectoryByGenreRow,
  TrajectoryOverallRow,
} from "./model/types";
import { renderTableHTML } from "../../shared/ui/renderTableHTML";
import type { TableColumn } from "../../shared/ui/tableColumn";
import type { SortDir } from "../../shared/lib/sortRows";

const TRAJECTORY_METRICS = [
  "content_distance",
  "structural_distance",
  "adjacent_similarity_distance",
  "step_magnitude_distance",
  "turning_angle_distance",
];

type RenderableRow = FacetableRow & TextVariantRow & NamedRow;

interface FamilyData {
  parallelism_overall: ParallelismOverallRow[];
  parallelism_by_type: ParallelismByTypeRow[];
  genre_overall: GenreOverallRow[];
  genre_by_genre: GenreByGenreRow[];
  trajectory: TrajectoryOverallRow[];
  trajectory_by_genre: TrajectoryByGenreRow[];
}

const EMPTY_FAMILY_DATA: FamilyData = {
  parallelism_overall: [],
  parallelism_by_type: [],
  genre_overall: [],
  genre_by_genre: [],
  trajectory: [],
  trajectory_by_genre: [],
};

declare const DATA: { families: Record<string, FamilyData> };

function familyData(id: string): FamilyData {
  return DATA.families[id] ?? EMPTY_FAMILY_DATA;
}

function sentenceCase(s: string): string {
  const spaced = s.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

interface TableState {
  scope: string;
  unit: string;
  text: string;
  filter: string;
  sortKey: string;
  sortDir: SortDir;
}

let currentFamily = "semantic";

const state: Record<
  "par-table" | "gen-table",
  TableState & { metric?: string }
> = {
  "par-table": {
    scope: "overall",
    unit: "all",
    text: "all",
    filter: "",
    sortKey: "separation_auc",
    sortDir: "desc",
  },
  "gen-table": {
    metric: "genre",
    scope: "overall",
    unit: "all",
    text: "all",
    filter: "",
    sortKey: "separation_auc",
    sortDir: "desc",
  },
};

function applyRowFilters<T extends RenderableRow>(
  rows: T[],
  s: TableState,
): T[] {
  let filtered = applyFacetFilter(rows, currentFamily, s.unit);
  filtered = applyTextFilter(filtered, s.text);
  filtered = applyNameFilter(filtered, s.filter);
  return filtered;
}

function attachSortHandlers(
  tableEl: HTMLTableElement,
  s: TableState,
  onSort: () => void,
): void {
  tableEl.querySelectorAll("thead th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = (th as HTMLElement).dataset.key as string;
      s.sortDir = key === s.sortKey && s.sortDir === "desc" ? "asc" : "desc";
      s.sortKey = key;
      onSort();
    });
  });
}

function renderInto<T extends RenderableRow>(
  tableEl: HTMLTableElement,
  s: TableState,
  rows: T[],
  columns: TableColumn<T>[],
  onSort: () => void,
): void {
  const filtered = applyRowFilters(rows, s);
  tableEl.innerHTML = renderTableHTML(filtered, columns, s.sortKey, s.sortDir);
  attachSortHandlers(tableEl, s, onSort);
}

function render(tableId: "par-table" | "gen-table"): void {
  const s = state[tableId];
  const tableEl = document.getElementById(tableId) as HTMLTableElement;
  const family = familyData(currentFamily);
  const onSort = (): void => render(tableId);

  if (tableId === "par-table") {
    if (s.scope === "overall") {
      renderInto(
        tableEl,
        s,
        family.parallelism_overall,
        parallelismOverallColumns(),
        onSort,
      );
    } else {
      const rows = family.parallelism_by_type.filter(
        (r) => r.scope === s.scope,
      );
      renderInto(tableEl, s, rows, parallelismByTypeColumns(), onSort);
    }
    return;
  }

  const scopeWrap = document.getElementById("gen-scope-wrap") as HTMLElement;
  scopeWrap.style.display = "";
  if (s.metric === "genre") {
    if (s.scope === "overall") {
      renderInto(
        tableEl,
        s,
        family.genre_overall,
        genreOverallColumns(),
        onSort,
      );
    } else {
      const rows = family.genre_by_genre.filter((r) => r.genre === s.scope);
      renderInto(tableEl, s, rows, genreByGenreColumns(), onSort);
    }
    return;
  }
  if (s.scope === "overall") {
    const rows = family.trajectory.filter((r) => r.metric === s.metric);
    renderInto(tableEl, s, rows, trajectoryOverallColumns(rows), onSort);
  } else {
    const rows = family.trajectory_by_genre.filter(
      (r) => r.metric === s.metric && r.genre === s.scope,
    );
    renderInto(tableEl, s, rows, trajectoryByGenreColumns(), onSort);
  }
}

/** Wires every control, tab, and table on the results page to the injected DATA global. */
export function initBenchmarkTables(): void {
  const familyTabsNav = document.getElementById("family-tabs") as HTMLElement;
  FAMILIES.forEach((family, i) => {
    const btn = document.createElement("button");
    btn.dataset.family = family.id;
    btn.textContent = family.label;
    if (i === 0) btn.classList.add("active");
    familyTabsNav.appendChild(btn);
  });

  const parScopeSel = document.getElementById("par-scope") as HTMLSelectElement;
  const PARALLELISM_TYPE_ORDER = [
    "Synonymous",
    "Antithetic",
    "Synthetic",
    "Emblematic",
    "Staircase",
  ];
  PARALLELISM_TYPE_ORDER.forEach((t) => {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = t;
    parScopeSel.appendChild(o);
  });
  parScopeSel.addEventListener("change", () => {
    state["par-table"].scope = parScopeSel.value;
    state["par-table"].sortKey = "separation_auc";
    render("par-table");
  });

  const parUnitSel = document.getElementById("par-unit") as HTMLSelectElement;
  const genUnitSel = document.getElementById("gen-unit") as HTMLSelectElement;
  const parUnitLabel = document.getElementById("par-unit-label") as HTMLElement;
  const genUnitLabel = document.getElementById("gen-unit-label") as HTMLElement;

  function populateFacetOptions(): void {
    const facet = FACETS[currentFamily];
    [parUnitSel, genUnitSel].forEach((sel) => {
      sel
        .querySelectorAll("option:not([value='all'])")
        .forEach((o) => o.remove());
      if (facet) {
        facet.values.forEach((v) => {
          const o = document.createElement("option");
          o.value = v;
          o.textContent = sentenceCase(v);
          sel.appendChild(o);
        });
      }
    });
    const label = facet ? facet.label : "Unit";
    parUnitLabel.textContent = label;
    genUnitLabel.textContent = label;
  }

  parUnitSel.addEventListener("change", () => {
    state["par-table"].unit = parUnitSel.value;
    updateTextFilterVisibility();
    render("par-table");
  });

  const parTextSel = document.getElementById("par-text") as HTMLSelectElement;
  ["consonantal", "vocalized", "cantillation"].forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = sentenceCase(v);
    parTextSel.appendChild(o);
  });
  parTextSel.addEventListener("change", () => {
    state["par-table"].text = parTextSel.value;
    render("par-table");
  });

  document.getElementById("par-filter")!.addEventListener("input", (e) => {
    state["par-table"].filter = (e.target as HTMLInputElement).value;
    render("par-table");
  });

  const genMetricSel = document.getElementById(
    "gen-metric",
  ) as HTMLSelectElement;
  const metricOption = (value: string, label: string): HTMLOptionElement => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    return o;
  };
  genMetricSel.appendChild(metricOption("genre", "Genre discrimination"));
  TRAJECTORY_METRICS.forEach((m) =>
    genMetricSel.appendChild(metricOption(m, sentenceCase(m))),
  );
  genMetricSel.addEventListener("change", () => {
    state["gen-table"].metric = genMetricSel.value;
    state["gen-table"].scope = "overall";
    (document.getElementById("gen-scope") as HTMLSelectElement).value =
      "overall";
    state["gen-table"].sortKey =
      state["gen-table"].metric === "genre"
        ? "separation_auc"
        : "raw_effect_size";
    render("gen-table");
  });

  const genScopeSel = document.getElementById("gen-scope") as HTMLSelectElement;
  function populateGenreScopeOptions(): void {
    genScopeSel
      .querySelectorAll("option:not([value='overall'])")
      .forEach((o) => o.remove());
    const genres = new Set(
      familyData(currentFamily).genre_by_genre.map((r) => r.genre),
    );
    [...genres].forEach((g) => {
      const o = document.createElement("option");
      o.value = g;
      o.textContent = g;
      genScopeSel.appendChild(o);
    });
  }
  populateGenreScopeOptions();
  genScopeSel.addEventListener("change", () => {
    const s = state["gen-table"];
    s.scope = genScopeSel.value;
    if (s.metric !== "genre") {
      s.sortKey = s.scope === "overall" ? "raw_effect_size" : "gap";
    }
    render("gen-table");
  });

  genUnitSel.addEventListener("change", () => {
    state["gen-table"].unit = genUnitSel.value;
    updateTextFilterVisibility();
    render("gen-table");
  });

  const genTextSel = document.getElementById("gen-text") as HTMLSelectElement;
  ["consonantal", "vocalized", "cantillation"].forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = sentenceCase(v);
    genTextSel.appendChild(o);
  });
  genTextSel.addEventListener("change", () => {
    state["gen-table"].text = genTextSel.value;
    render("gen-table");
  });

  document.getElementById("gen-filter")!.addEventListener("input", (e) => {
    state["gen-table"].filter = (e.target as HTMLInputElement).value;
    render("gen-table");
  });

  document.querySelectorAll("nav.tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll("nav.tabs button")
        .forEach((b) => b.classList.remove("active"));
      document
        .querySelectorAll("section.tab")
        .forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      document
        .getElementById("tab-" + (btn as HTMLElement).dataset.tab)!
        .classList.add("active");
    });
  });

  const parTextWrap = document.getElementById("par-text-wrap") as HTMLElement;
  const genTextWrap = document.getElementById("gen-text-wrap") as HTMLElement;
  const parUnitWrap = document.getElementById("par-unit-wrap") as HTMLElement;
  const genUnitWrap = document.getElementById("gen-unit-wrap") as HTMLElement;

  function updateTextFilterVisibility(): void {
    const isLexical = currentFamily === "lexical";
    const hasFacet = !!FACETS[currentFamily];
    parUnitWrap.style.display = hasFacet ? "" : "none";
    genUnitWrap.style.display = hasFacet ? "" : "none";

    const family = familyData(currentFamily);
    const parHasText = isLexical
      ? state["par-table"].unit === "word"
      : familyHasTextVariant(family.parallelism_overall);
    const genHasText = isLexical
      ? state["gen-table"].unit === "word"
      : familyHasTextVariant(family.genre_overall);
    parTextWrap.style.display = parHasText ? "" : "none";
    genTextWrap.style.display = genHasText ? "" : "none";
    if (!parHasText) {
      state["par-table"].text = "all";
      parTextSel.value = "all";
    }
    if (!genHasText) {
      state["gen-table"].text = "all";
      genTextSel.value = "all";
    }
  }

  document.querySelectorAll("nav.family-tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll("nav.family-tabs button")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentFamily = (btn as HTMLElement).dataset.family as string;
      populateFacetOptions();
      state["par-table"].unit = "all";
      parUnitSel.value = "all";
      state["gen-table"].unit = "all";
      genUnitSel.value = "all";
      populateGenreScopeOptions();
      if (
        ![...genScopeSel.options].some(
          (o) => o.value === state["gen-table"].scope,
        )
      ) {
        state["gen-table"].scope = "overall";
        genScopeSel.value = "overall";
      }
      updateTextFilterVisibility();
      render("par-table");
      render("gen-table");
    });
  });

  updateTextFilterVisibility();
  render("par-table");
  render("gen-table");
}
