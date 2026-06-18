# SDL_agent - Software Module (v2.6.1)

## Software API Endpoints

### Phase 1: CSV Preview & Inspection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/csv/preview?path=<abs>&n=20` | GET | CSV preview with column type inference |
| `/api/csv/columns?path=<abs>` | GET | Lightweight column name list only |

**`/api/csv/preview` response:**
```json
{
  "success": true,
  "data": {
    "path": "...",
    "columns": [{"name": "PCE", "type": "float", "sample": ["15.2","16.1"], "non_null_count": 5, "null_count": 0}],
    "row_count": 20,
    "total_rows": 1234,
    "file_size": 98765
  }
}
```

**Files:** `software/csv_inspector.py` (inspect_csv), `frontend/src/api/analysis.ts` (previewCsv, getCsvColumns)

### Phase 2: Result Schema Convention

Each algorithm can declare a `result_schema` on `BaseAlgorithm` class:

```python
class MyAlgo(BaseAlgorithm):
    name = "my_algo"
    result_schema = {
        "type": "table",          # table | kv | chart | matrix | list
        "sections": [{
            "title": "Statistics",
            "columns": [{"key": "mean", "label": "Mean", "format": "decimal:3"}],
            "rows_from": "statistics"
        }]
    }
```

**Frontend rendering chain:**
`ResultRenderer` -> dispatches by `schema.type` ->
- `table` -> `ResultTable` / `ResultMatrix` (if subsection type=matrix)
- `kv` -> `ResultKvList`
- `chart` -> `ResultChart`
- `matrix` -> `ResultMatrix`
- Fallback: `<pre>` JSON dump

**Types:** `frontend/src/types/analysis.ts` - `ResultSchema`, `TableColumn`, `ResultSection`, `KvItem`, `ChartConfig`

**Modified:** `BaseAlgorithm.result_schema` field, 4 algorithms declare schema (data_statistics, data_normalization, spectrum_analysis, bayesian_optimization)

### Phase 3: Algorithm Recommendation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/algorithm/recommend` | POST | LLM recommends best algorithm + read function for a CSV |

**Request:** `{"path": "temporal/extraction.csv"}`
**Response:**
```json
{
  "success": true,
  "data": {
    "algorithm": "spectrum_analysis",
    "read_function": "read_numeric_columns",
    "read_params": {},
    "reasoning": "Detected wavelength and intensity columns..."
  }
}
```

**Files:**
- `software/auto_analyze.py` — `recommend_algorithm()` sync function (recommend only, no execution)
- `software/auto_analyze.py` — `run_pipeline()` async with callbacks (full auto-analysis)
- `frontend/src/components/result/RecommendModal.vue` — modal with Apply/Try-Another
- `frontend/src/pages/AnalysisPage.vue` — Sparkles button in run-bar

### Frontend Architecture

**Stores:** `stores/analysis.ts` (useAnalysisStore) — algorithms, csvFiles, previewCache, recommend state

**Components:**
- `components/result/ParamForm.vue` — dynamic form from params_schema (bool/int/float/str/list/columns)
- `components/result/ResultRenderer.vue` — dispatcher by result_schema.type
- `components/result/ResultTable.vue` — formatted table (decimal/percent/scientific)
- `components/result/ResultKvList.vue` — key-value list with units
- `components/result/ResultMatrix.vue` — correlation matrix heatmap
- `components/result/ResultChart.vue` — uPlot line/bar chart
- `components/result/RecommendModal.vue` — smart algorithm recommendation modal
- `components/cards/ResultCard.vue` — fallback card wrapper (no-schema algorithms)
- `components/modals/FileSelectorModal.vue` — file/dir picker with CSV inline preview

### i18n Conventions

All UI text uses `$t('analysis.xxx')` keys. Source: `frontend/src/locales/en.json`. Add new keys there first, then run:
```bash
python utils/i18n/sync.py    # sync en->zh [待翻译] markers
python utils/i18n/translate.py  # LLM auto-translate (requires API key)
python utils/i18n/check.py   # verify alignment (exit 1 on mismatch)
```

### Testing
```bash
pytest platform_init/test/software/ -x -v
# 72 tests covering:
#   - csv_inspector (column type inference, preview API)
#   - algorithm result_schema (BaseAlgorithm field, list_algorithms passthrough)
#   - software direct (controller, manager, auto_analyze, generate, reload)
#   - software in experiment (compile, execute, backcompat)
```
