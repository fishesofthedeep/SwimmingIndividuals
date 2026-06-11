# ===================================================================
# asc_environment.jl
# -------------------------------------------------------------------
# Time-varying, environmentally-driven forcing from ESRI ASCII (.asc) rasters,
# catalogued in a small model-native XML (STConfig.xml).
#
# This module is ADDITIVE: the existing NetCDF path (generate_environment!,
# initial_habitat_capacity) still works unchanged. When an `env_xml` entry is
# present in files.csv, setup wires an `EnvForcing` object into the module-level
# `ENV_FORCING` Ref (same pattern as RUNTIME_CONFIG, so the MarineModel struct
# layout is untouched). `update_environment_from_asc!` is then called from the
# month-advance hook in TimeStep! to:
#
#   1. load the monthly driver layers (tos/tob/sos/sob) for the CURRENT absolute
#      model date and refresh `model.environment.data[...]` used by energetics;
#   2. recompute the focal + resource habitat-capacity slice for the current
#      calendar month from the driver envelope (so capacity genuinely tracks SST
#      and shifts north in summer / south in winter); and
#   3. expose a normalised primary-production multiplier (npp) for the resource
#      growth function.
#
# CONFIG SCHEMA (STConfig.xml) — one entry per variable, directory + monthly
# filename pattern, plus a single time span. No vendor/plugin metadata:
#
#   <EnvironmentConfig name="...">
#     <TimeSpan start="YYYY-MM" end="YYYY-MM"/>
#     <Variable name="tos" role="driver"/>           <!-- surface temperature   -->
#     <Variable name="tob" role="driver"/>           <!-- bottom temperature    -->
#     <Variable name="sos" role="driver"/>           <!-- surface salinity      -->
#     <Variable name="sob" role="driver"/>           <!-- bottom salinity       -->
#     <Variable name="npp" role="relpp"/>            <!-- relative prim. prod.  -->
#   </EnvironmentConfig>
#
# Per <Variable>: role is "driver" (feeds the habitat-capacity envelope), "relpp"
# (npp -> resource carrying capacity), or "habitat" (direct capacity input).
# Optional attributes: dir="<subfolder>" (default = name) and
# file="<pattern>" (default "<name>_{Y}-{M}.asc", where {Y}=4-digit year,
# {M}=2-digit month). Files are resolved as  <env_dir>/<dir>/<file>, with
# <env_dir> taken from the files.csv `env_dir` row (e.g. EnvData). A variable may
# instead carry explicit <File Date="YYYY-MM-DD">name.asc</File> children if a
# non-uniform series is needed.
#
# All five variables form a SINGLE monthly set used to build monthly habitat
# capacities. If the run extends past the last available month, the final year is
# repeated (see `_entry_for`): a date beyond `end` reuses the same calendar month
# of the final year, and a date before `start` reuses the first year.
# ===================================================================

using Dates  # (re-import is a no-op; module already imports Dates/DelimitedFiles)

# --- Module-level handle (mirrors RUNTIME_CONFIG pattern in timestep.jl) -------
const ENV_FORCING = Ref{Any}(nothing)

# -------------------------------------------------------------------
# Parsed catalogue types
# -------------------------------------------------------------------
struct AscEntry
    date::Date
    path::String      # resolved, OS-appropriate path on the machine running the model
end

mutable struct AscVariable
    name::String                 # canonical variable, e.g. "tos", "npp"
    role::String                 # "driver" | "relpp" | "habitat"
    entries::Vector{AscEntry}    # sorted ascending by date
    available::Union{Nothing, Vector{AscEntry}}   # cached subset whose files exist on disk
end

mutable struct EnvForcing
    vars::Dict{String, AscVariable}           # canonical name => AscVariable
    # model-grid geometry (cell centres), filled from grid.csv
    lonres::Int
    latres::Int
    lon_centers::Vector{Float64}              # length lonres (index 1 = west)
    lat_centers::Vector{Float64}              # length latres (index 1 = south, matches placement)
    # caches keyed by (canonical_name, Date) => model-grid Matrix{Float32}
    cache::Dict{Tuple{String, Date}, Matrix{Float32}}
    # reference NPP map (mean over available npp files) for normalising the multiplier
    npp_reference::Union{Nothing, Matrix{Float32}}
    nodata_fill::Float32                      # value used where a layer has NODATA / ocean-outside
    ocean_mask::Union{Nothing, Matrix{Bool}}  # true = ocean; false cells force capacity 0 (land)
end

# -------------------------------------------------------------------
# Robust ESRI ASCII reader (returns header dict + raw matrix as stored,
# i.e. row 1 == NORTH-most row per the ESRI convention).
# -------------------------------------------------------------------
function read_asc_with_header(path::String)
    open(path, "r") do f
        hdr = Dict{String, Float64}()
        # ESRI headers are up to 6 lines: ncols, nrows, xllcorner|xllcenter,
        # yllcorner|yllcenter, cellsize, NODATA_value. A HEADER line's first token
        # is a keyword that does NOT parse as a number (e.g. "ncols",
        # "NODATA_value"); a DATA line's first token parses as a number — including
        # "nan"/"inf", which is why we must test numeric-parse rather than
        # "is it alphabetic" (NODATA cells are written as "nan" and would otherwise
        # be mistaken for header keywords).
        known = Set(["ncols","nrows","xllcorner","yllcorner","xllcenter",
                     "yllcenter","cellsize","nodata_value"])
        pos = position(f)
        while !eof(f)
            line = readline(f)
            toks = split(strip(line))
            if length(toks) < 2 || tryparse(Float64, toks[1]) !== nothing
                seek(f, pos); break          # first token is numeric -> data row
            end
            key = lowercase(toks[1])
            (key in known) || (seek(f, pos); break)   # unknown keyword -> stop
            val = tryparse(Float64, toks[2])
            val === nothing && (seek(f, pos); break)
            hdr[key] = val
            pos = position(f)
            length(hdr) >= 6 && break
        end
        data = readdlm(f)
        return hdr, Matrix{Float64}(data)
    end
end

# Cell-centre coordinate arrays for an ESRI grid from its header.
function asc_cell_centers(hdr::Dict{String,Float64})
    ncols = Int(hdr["ncols"]); nrows = Int(hdr["nrows"]); cs = hdr["cellsize"]
    xll = haskey(hdr, "xllcorner") ? hdr["xllcorner"] : (hdr["xllcenter"] - cs/2)
    yll = haskey(hdr, "yllcorner") ? hdr["yllcorner"] : (hdr["yllcenter"] - cs/2)
    lon_c = [xll + (c - 0.5)*cs for c in 1:ncols]                 # west -> east
    # ESRI row 1 is the NORTH-most row; convert to geographic latitude per row.
    lat_row = [yll + (nrows - r + 0.5)*cs for r in 1:nrows]       # row 1 -> north
    nodata = get(hdr, "nodata_value", -9999.0)
    return lon_c, lat_row, Float32(nodata)
end

# -------------------------------------------------------------------
# Resample an .asc onto the model grid by nearest geographic cell centre.
# This is orientation-safe: it never assumes the .asc row order matches the
# model array; it matches on lon/lat coordinates directly. NODATA cells map to
# `nodata_fill` (NaN by default) so habitat code can treat them as land/ocean-out.
# -------------------------------------------------------------------
function regrid_to_model(hdr, raw::Matrix{Float64}, f::EnvForcing)
    lon_c, lat_row, nodata = asc_cell_centers(hdr)
    nrows = length(lat_row); ncols = length(lon_c)
    out = fill(f.nodata_fill, f.lonres, f.latres)

    @inbounds for j in 1:f.latres
        # nearest asc row to this model latitude
        latm = f.lat_centers[j]
        r = searchsortednearest(lat_row, latm)               # lat_row is descending
        for i in 1:f.lonres
            lonm = f.lon_centers[i]
            c = searchsortednearest(lon_c, lonm)             # lon_c is ascending
            v = raw[r, c]
            if isfinite(v) && v != nodata
                out[i, j] = Float32(v)
            end
        end
    end
    return out
end

# nearest index in a monotonically (ascending or descending) ordered vector
@inline function searchsortednearest(v::AbstractVector, x)
    n = length(v)
    n == 1 && return 1
    ascending = v[end] >= v[1]
    # linear scan is fine for the modest grid sizes here (≤ a few hundred);
    # avoids edge cases with searchsorted on descending vectors.
    best = 1; bestd = abs(v[1] - x)
    @inbounds for k in 2:n
        d = abs(v[k] - x)
        if d < bestd; bestd = d; best = k; end
    end
    return best
end

# -------------------------------------------------------------------
# XML parsing — dependency-free (no EzXML required). The config is small,
# well-formed XML; fields are pulled with targeted regexes. If EzXML is ever
# added to the project this could be swapped, but staying dependency-free keeps
# Project.toml untouched.
# -------------------------------------------------------------------
function _role_from_attr(role::AbstractString)
    r = lowercase(strip(String(role)))
    (r == "relpp" || r == "rel_pp" || r == "pp")     && return "relpp"
    (r == "habitat" || r == "habitatcapacity")        && return "habitat"
    (r == "static" || r == "bathymetry" || r == "depth") && return "static"
    return "driver"
end

# Resolve a variable's monthly file path for a given date from its directory and
# filename pattern, rooted at base_dir (the files.csv `env_dir`). Pattern tokens:
#   {Y} 4-digit year, {M} 2-digit month, {y}/{m} non-padded.
function _pattern_path(base_dir::AbstractString, dir::AbstractString,
                       pattern::AbstractString, d::Date)
    Y = string(year(d)); M = lpad(month(d), 2, '0')
    fname = replace(pattern, "{Y}"=>Y, "{M}"=>M, "{y}"=>string(year(d)), "{m}"=>string(month(d)))
    if !isempty(base_dir)
        return joinpath(base_dir, dir, fname)
    else
        return joinpath(dir, fname)   # relative to cwd if no env_dir given
    end
end

# Parse "YYYY-MM" (or "YYYY-MM-DD") into the first-of-month Date.
function _parse_ym(s::AbstractString)
    s = strip(String(s))
    m = match(r"^(\d{4})-(\d{1,2})", s)
    m === nothing && return nothing
    return Date(parse(Int, m.captures[1]), parse(Int, m.captures[2]), 1)
end

"""
    parse_env_config(xml_path; base_dir="")

Parse the model-native STConfig.xml into a Dict{String,AscVariable}. Each
`<Variable name=.. role=.. [dir=..] [file=..]>` becomes one variable whose monthly
file series is generated from the `<TimeSpan start=.. end=..>` using the filename
pattern (default `"<name>_{Y}-{M}.asc"`, subfolder default `<name>`), or from
explicit `<File Date=..>name.asc</File>` children if present. `base_dir` is the
files.csv `env_dir` (e.g. EnvData) under which the per-variable folders live.
"""
function parse_env_config(xml_path::String; base_dir::AbstractString="")
    text = read(xml_path, String)
    # Strip XML comments first, so example tags inside <!-- ... --> can never be
    # picked up by the element regexes below.
    text = replace(text, r"<!--.*?-->"s => "")
    vars = Dict{String, AscVariable}()

    # Optional global time span.
    tsm = match(r"<TimeSpan\b[^>]*>"s, text)
    span_start = nothing; span_end = nothing
    if tsm !== nothing
        sm = match(r"start=\"(.*?)\"", tsm.match); em = match(r"end=\"(.*?)\"", tsm.match)
        sm !== nothing && (span_start = _parse_ym(sm.captures[1]))
        em !== nothing && (span_end   = _parse_ym(em.captures[1]))
    end

    # Each <Variable .../> or <Variable ...>...</Variable>.
    for vm in eachmatch(r"<Variable\b([^>]*?)(/>|>(.*?)</Variable>)"s, text)
        attrs = vm.captures[1]; inner = vm.captures[3] === nothing ? "" : vm.captures[3]
        nm = match(r"name=\"(.*?)\"", attrs); nm === nothing && continue
        name = lowercase(strip(nm.captures[1]))
        rm   = match(r"role=\"(.*?)\"", attrs)
        role = rm === nothing ? "driver" : _role_from_attr(rm.captures[1])
        dm   = match(r"dir=\"(.*?)\"", attrs)
        dir  = dm === nothing ? name : strip(dm.captures[1])
        fm   = match(r"file=\"(.*?)\"", attrs)
        pattern = fm === nothing ? "$(name)_{Y}-{M}.asc" : strip(fm.captures[1])

        av = get!(vars, name, AscVariable(name, role, AscEntry[], nothing))

        # Explicit <File> children take precedence if given.
        files_listed = false
        for fmm in eachmatch(r"<File\s+Date=\"(.*?)\"\s*>(.*?)</File>"s, inner)
            files_listed = true
            d = _parse_ym(fmm.captures[1]); d === nothing && continue
            fpath = isempty(base_dir) ? joinpath(dir, strip(fmm.captures[2])) :
                                        joinpath(base_dir, dir, strip(fmm.captures[2]))
            push!(av.entries, AscEntry(d, fpath))
        end

        # Otherwise generate the file series.
        has_date_token = occursin("{Y}", pattern) || occursin("{M}", pattern) ||
                         occursin("{y}", pattern) || occursin("{m}", pattern)
        if !files_listed && !has_date_token
            # STATIC variable (e.g. bathymetry): one time-invariant raster used for
            # every date. Resolved as <env_dir>/<dir>/<pattern>. dir defaults to the
            # variable name, so for a file sitting directly in EnvData use dir=".".
            spath = isempty(base_dir) ? joinpath(dir, pattern) : joinpath(base_dir, dir, pattern)
            sdate = span_start === nothing ? Date(1, 1, 1) : span_start
            push!(av.entries, AscEntry(sdate, spath))
        elseif !files_listed && span_start !== nothing && span_end !== nothing
            # Monthly series across the time span.
            d = span_start
            while d <= span_end
                push!(av.entries, AscEntry(d, _pattern_path(base_dir, dir, pattern, d)))
                d += Month(1)
            end
        end
    end

    for (_, av) in vars
        sort!(av.entries, by = e -> e.date)
        last_for = Dict{Date,AscEntry}()
        for e in av.entries; last_for[e.date] = e; end
        av.entries = [last_for[d] for d in sort(collect(keys(last_for)))]
    end
    return vars
end

# -------------------------------------------------------------------
# Build the EnvForcing object (grid geometry from grid.csv-style DataFrame).
# -------------------------------------------------------------------
"""
    load_env_forcing(xml_path, grid; base_dir="", nodata_fill=NaN32)

Construct an `EnvForcing` from the model-native STConfig.xml and the model `grid`
DataFrame (`Name`/`Value` columns from grid.csv). `base_dir` is the files.csv
`env_dir` under which the per-variable `.asc` folders live.
"""
function load_env_forcing(xml_path::String, grid; base_dir::AbstractString="",
                          nodata_fill::Float32=NaN32)
    g(name) = grid[grid.Name .== name, :Value][1]
    lonres = Int(g("lonres")); latres = Int(g("latres"))
    cs   = Float64(g("cellsize"))
    xll  = Float64(g("xllcorner"))
    yll  = Float64(g("yllcorner"))
    lon_centers = [xll + (i - 0.5)*cs for i in 1:lonres]          # west -> east
    lat_centers = [yll + (j - 0.5)*cs for j in 1:latres]          # south -> north (matches placement: y=1 south)

    vars = parse_env_config(xml_path; base_dir=base_dir)
    f = EnvForcing(vars, lonres, latres, lon_centers, lat_centers,
                   Dict{Tuple{String,Date},Matrix{Float32}}(), nothing, nodata_fill, nothing)

    ntot = sum(length(v.entries) for (_, v) in vars; init=0)
    @info "Loaded environmental config: $(length(vars)) variables, $ntot monthly layers ($(join(sort(collect(keys(vars))), ", ")))."
    return f
end

# Pick the catalogue entry for `date`. Within the available span this is the exact
# (year,month). PAST THE END the FINAL YEAR IS REPEATED — a date later than the
# last available month reuses the same calendar month of the final year; a date
# before the first month reuses the first year. (This is the "repeat final year"
# behaviour for runs that extend past the forcing data.)
# Choose the entry for `date` from a given (ascending, non-empty) entry list.
# Exact (year,month) when present; otherwise REPEAT THE FINAL YEAR past the end
# (same calendar month of the last year) and the first year before the start, so
# the seasonal cycle keeps looping when we run out of data. Used both for the full
# catalogue and for the on-disk-available subset.
function _select_entry(entries::Vector{AscEntry}, date::Date)
    isempty(entries) && return nothing
    ym = (year(date), month(date))
    for e in entries
        (year(e.date), month(e.date)) == ym && return e
    end
    first_year = year(entries[1].date)
    last_year  = year(entries[end].date)
    target_year = year(date) > last_year  ? last_year  :
                  year(date) < first_year ? first_year : year(date)
    # same calendar month within the chosen (clamped) year -> preserves seasonality
    for e in entries
        (year(e.date), month(e.date)) == (target_year, month(date)) && return e
    end
    # same month, nearest available year
    same_month = filter(e -> month(e.date) == month(date), entries)
    if !isempty(same_month)
        return same_month[argmin(abs.(year.(getfield.(same_month, :date)) .- target_year))]
    end
    # last resort: nearest date overall
    return entries[argmin(abs.(Dates.value.(getfield.(entries, :date) .- date)))]
end

function _entry_for(av::AscVariable, date::Date)
    return _select_entry(av.entries, date)
end

# The subset of this variable's entries whose .asc files actually exist on disk
# (computed once, then cached). Lets us replay the last AVAILABLE year when the
# catalogue claims files that are not present (data ends early / has gaps).
function _available_entries(av::AscVariable)
    if av.available === nothing
        av.available = filter(e -> isfile(e.path), av.entries)
    end
    return av.available
end

"""
    env_layer(f, canonical_name, date) -> Matrix{Float32} | nothing

Return the model-grid layer for a canonical variable at `date` (cached).
"""
function env_layer(f::EnvForcing, name::AbstractString, date::Date)
    av = get(f.vars, lowercase(String(name)), nothing)
    av === nothing && return nothing
    e = _entry_for(av, date)
    e === nothing && return nothing
    key = (av.name, e.date)
    haskey(f.cache, key) && return f.cache[key]

    # If the catalogue's chosen file is not on disk, the data has run out (or has a
    # gap). Replay the most recent AVAILABLE file of the same calendar month so the
    # seasonal cycle keeps going instead of going stale.
    if !isfile(e.path)
        avail = _available_entries(av)
        if isempty(avail)
            @warn "No $(av.name) .asc files found on disk; cannot force this variable."
            return nothing
        end
        e = _select_entry(avail, date)
        e === nothing && return nothing
        key = (av.name, e.date)
        haskey(f.cache, key) && return f.cache[key]
        @info "$(av.name): no file for $(year(date))-$(lpad(month(date),2,'0')); replaying last available same-month layer ($(e.date))."
    end

    hdr, raw = read_asc_with_header(e.path)
    grid_layer = regrid_to_model(hdr, raw, f)
    f.cache[key] = grid_layer
    return grid_layer
end

# -------------------------------------------------------------------
# Primary-production multiplier (relative PP -> resource carrying capacity).
#
# Rationale (literature): ecosystem models commonly let a spatial-temporal
# "relative primary production" layer scale the local production/biomass of
# producer groups (Christensen & Walters 2004, Ecol. Modelling; Steenbeek et al.
# 2013, Ecol. Modelling — spatial-temporal data framework). Productivity sets the
# standing-stock the system can support, so we apply NPP as a multiplier on the
# resource CARRYING CAPACITY K (not the intrinsic rate r): doubling productivity
# doubles the equilibrium biomass a cell can sustain, whereas scaling r would only
# change how fast K is approached. The multiplier is normalised to the long-term
# mean NPP field so that a value of 1 reproduces the baseline K from the trait
# table; cells/months above (below) average productivity get K scaled up (down).
# -------------------------------------------------------------------
function _npp_reference!(f::EnvForcing)
    f.npp_reference !== nothing && return f.npp_reference
    av = get(f.vars, "npp", nothing)
    av === nothing && return nothing
    acc = zeros(Float64, f.lonres, f.latres); cnt = 0
    # average over a representative span (cap at 120 layers for speed)
    step = max(1, fld(length(av.entries), 120))
    for k in 1:step:length(av.entries)
        e = av.entries[k]
        isfile(e.path) || continue
        hdr, raw = read_asc_with_header(e.path)
        layer = regrid_to_model(hdr, raw, f)
        @inbounds for idx in eachindex(layer)
            v = layer[idx]
            if isfinite(v); acc[idx] += v; end
        end
        cnt += 1
    end
    cnt == 0 && return nothing
    ref = Float32.(acc ./ cnt)
    # guard against zeros in the reference (avoids divide-by-zero downstream)
    gmean = Float32(max(1f-12, sum(filter(isfinite, ref)) / max(1, count(isfinite, ref))))
    @inbounds for idx in eachindex(ref)
        (!isfinite(ref[idx]) || ref[idx] <= 0f0) && (ref[idx] = gmean)
    end
    f.npp_reference = ref
    return ref
end

"""
    npp_multiplier(f, date; clamp_range=(0.2f0, 5.0f0)) -> Matrix{Float32} (lon×lat)

Normalised NPP multiplier for `date`: current NPP divided by the long-term mean
NPP field, clamped to a sane range. Returns an all-ones matrix if no npp layers
are present (i.e. NPP forcing simply disabled).
"""
function npp_multiplier(f::EnvForcing, date::Date; clamp_range=(0.2f0, 5.0f0))
    ones_grid = ones(Float32, f.lonres, f.latres)
    cur = env_layer(f, "npp", date)
    cur === nothing && return ones_grid
    ref = _npp_reference!(f)
    ref === nothing && return ones_grid
    out = ones_grid
    @inbounds for idx in eachindex(out)
        c = cur[idx]; r = ref[idx]
        if isfinite(c) && isfinite(r) && r > 0f0
            out[idx] = clamp(c / r, clamp_range[1], clamp_range[2])
        end
    end
    return out
end

# -------------------------------------------------------------------
# Habitat capacity from the SST envelope (and any other driver in envi_pref).
# Recomputes the CURRENT calendar-month slice of `capacities` in place from the
# absolute month's driver layers. Uses exactly the same trapezoidal preference
# envelope as initial_habitat_capacity (pref_min/opt_min/opt_max/pref_max) so the
# behaviour is consistent; the only change is that the driver values are now the
# true time-varying .asc layers instead of a 12-month climatology, so the capacity
# field shifts north/south as SST does.
#
# `prefs_df` columns: species, variable, pref_min, opt_min, opt_max, pref_max
# `spec_names`     : focal (1:n_spec) then resource SpeciesLong, matching the
#                    species axis of `capacities`.
# `var_alias`      : maps an envi_pref `variable` name to a canonical .asc name,
#                    e.g. "temp-surf" => "tos", "temp-bottom" => "tob",
#                    "sal-surf" => "sos". Anything already canonical passes through.
# -------------------------------------------------------------------
function _suitability(val, pmin, omin, omax, pmax)
    (ismissing(val) || !isfinite(val)) && return 0.0f0
    if any(ismissing, (pmin, omin, omax, pmax)); return 1.0f0; end
    if val >= omin && val <= omax
        return 1.0f0
    elseif val > pmin && val < omin
        return Float32((val - pmin) / (omin - pmin))
    elseif val > omax && val < pmax
        return Float32((pmax - val) / (pmax - omax))
    end
    return 0.0f0
end

# Ocean/land mask from a static `bathymetry` variable (true = ocean). Polarity is
# auto-detected from the domain-center cell (deep water): if center bathymetry is
# negative the field is elevation (ocean = value < 0); if positive it is depth
# (ocean = value > 0). NODATA/NaN is always land. Cached on the forcing object.
function ocean_mask_for(f::EnvForcing)
    f.ocean_mask !== nothing && return f.ocean_mask
    haskey(f.vars, "bathymetry") || return nothing
    bath = env_layer(f, "bathymetry", Date(1, 1, 1))
    bath === nothing && return nothing
    ci = max(1, f.lonres ÷ 2); cj = max(1, f.latres ÷ 2)
    cvals = Float64[]
    for dj in -2:2, di in -2:2
        ii = ci + di; jj = cj + dj
        if 1 <= ii <= f.lonres && 1 <= jj <= f.latres && isfinite(bath[ii, jj])
            push!(cvals, Float64(bath[ii, jj]))
        end
    end
    center = isempty(cvals) ? -1.0 : sort(cvals)[cld(length(cvals), 2)]
    ocean_is_negative = center < 0
    mask = falses(f.lonres, f.latres)
    @inbounds for j in 1:f.latres, i in 1:f.lonres
        v = bath[i, j]
        if isfinite(v)
            mask[i, j] = ocean_is_negative ? (v < 0f0) : (v > 0f0)
        end
    end
    @info "Ocean mask built from bathymetry ($(count(mask)) ocean / $(length(mask)) cells; ocean = bathymetry $(ocean_is_negative ? "< 0" : "> 0"))."
    f.ocean_mask = mask
    return mask
end

function update_capacity_from_drivers!(capacities_cpu::Array{Float32,4}, f::EnvForcing,
                                       date::Date, prefs_df, spec_names::Vector{String},
                                       var_alias::Dict{String,String})
    cal_month = month(date)
    lonres, latres, nmonths, nspec_tot = size(capacities_cpu)
    @assert cal_month <= nmonths

    # Pre-load the driver layers we might need (unique aliased canonical names).
    needed = unique(String[ get(var_alias, String(v), lowercase(String(v))) for v in prefs_df.variable ])
    layers = Dict{String, Union{Nothing,Matrix{Float32}}}()
    for cn in needed
        layers[cn] = env_layer(f, cn, date)
    end

    omask = ocean_mask_for(f)   # nothing if no bathymetry variable

    for i in 1:min(length(spec_names), nspec_tot)
        sp_prefs = filter(row -> row.species == spec_names[i], prefs_df)
        isempty(sp_prefs) && continue
        @inbounds for lat in 1:latres, lon in 1:lonres
            if omask !== nothing && !omask[lon, lat]
                capacities_cpu[lon, lat, cal_month, i] = 0.0f0
                continue
            end
            suit = 1.0f0
            for pr in eachrow(sp_prefs)
                cn = get(var_alias, String(pr.variable), lowercase(String(pr.variable)))
                lay = get(layers, cn, nothing)
                lay === nothing && continue          # driver not available -> ignore this axis
                val = lay[lon, lat]
                suit *= _suitability(val, pr.pref_min, pr.opt_min, pr.opt_max, pr.pref_max)
                suit <= 0f0 && break
            end
            capacities_cpu[lon, lat, cal_month, i] = suit
        end
    end
    return capacities_cpu
end

# -------------------------------------------------------------------
# Top-level per-month update, called from TimeStep! when the month advances.
# -------------------------------------------------------------------
"""
    update_environment_from_asc!(model, current_date; var_alias=default_alias())

When ASC forcing is active (ENV_FORCING set), refresh:
  * model.environment.data["temp"]  (4D lon,lat,depth,month) surface-temp broadcast
    over depth for the current calendar month (used by individual_temp!/energetics),
  * model.capacities current-month slice from the SST (+driver) envelope,
and cache the NPP multiplier for resource_growth! (stored on the forcing object).

Safe no-op if ENV_FORCING is unset.
"""
function default_var_alias()
    Dict(
        "temp"        => "tos",   # surface temperature drivers
        "temp-surf"   => "tos",
        "temp_surf"   => "tos",
        "sst"         => "tos",
        "temp-bottom" => "tob",
        "temp_bot"    => "tob",
        "sal-surf"    => "sos",
        "sal_surf"    => "sos",
        "salinity"    => "sos",
        "sal-bottom"  => "sob",
    )
end

function update_environment_from_asc!(model, current_date::Date;
                                      var_alias::Dict{String,String}=default_var_alias())
    f = ENV_FORCING[]
    f === nothing && return nothing
    arch = model.arch
    files = model.files
    grid = model.depths.grid

    g(name) = grid[grid.Name .== name, :Value][1]
    lonres = Int(g("lonres")); latres = Int(g("latres")); depthres = Int(g("depthres"))

    # --- 1. Refresh the temperature field used by energetics -------------------
    # individual_temp! reads envi.data["temp"] as (lon,lat,depth,month). We supply
    # surface temp (tos) broadcast down the column for the current calendar month.
    tos = env_layer(f, "tos", current_date)
    if tos !== nothing && haskey(model.environment.data, "temp")
        temp_dev = model.environment.data["temp"]
        tdims = size(temp_dev)
        if length(tdims) == 4 && tdims[1] == lonres && tdims[2] == latres
            cal_month = month(current_date)
            temp_cpu = Array(temp_dev)
            mfill = Float32(_nanmean(tos))
            @inbounds for z in 1:size(temp_cpu,3), lat in 1:latres, lon in 1:lonres
                v = tos[lon, lat]
                temp_cpu[lon, lat, z, cal_month] = isfinite(v) ? v : mfill
            end
            copyto!(temp_dev, array_type(arch)(temp_cpu))
        end
    end

    # --- 2. Recompute the habitat-capacity slice for the current month ---------
    prefs_df = CSV.read(files[files.File .== "envi_pref", :Destination][1], DataFrame)
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait", :Destination][1], DataFrame))))
    resource = Dict(pairs(eachcol(CSV.read(files[files.File .== "resource_trait", :Destination][1], DataFrame))))
    n_spec = model.n_species; n_res = model.n_resource
    spec_names = String.(vcat(trait[:SpeciesLong][1:n_spec], resource[:SpeciesLong][1:n_res]))

    cap_cpu = Array{Float32,4}(Array(model.capacities))
    update_capacity_from_drivers!(cap_cpu, f, current_date, prefs_df, spec_names, var_alias)
    copyto!(model.capacities, array_type(arch)(cap_cpu))

    # --- 3. Cache NPP multiplier for resource_growth! --------------------------
    f.cache[("__npp_mult__", Date(year(current_date), month(current_date), 1))] =
        npp_multiplier(f, current_date)

    return nothing
end

# helper: mean of finite entries
function _nanmean(A)
    s = 0.0; n = 0
    @inbounds for v in A
        if isfinite(v); s += v; n += 1; end
    end
    return n == 0 ? 0.0 : s / n
end

"""
    current_npp_multiplier(model, current_date) -> Matrix{Float32} (lon×lat) | nothing

Convenience accessor used by resource_growth!. Returns the cached NPP multiplier
for the current month (computing it if necessary), or `nothing` if ASC forcing /
NPP is not active so the caller can skip the NPP scaling entirely.
"""
function current_npp_multiplier(model, current_date::Date)
    f = ENV_FORCING[]
    f === nothing && return nothing
    haskey(f.vars, "npp") || return nothing
    key = ("__npp_mult__", Date(year(current_date), month(current_date), 1))
    haskey(f.cache, key) && return f.cache[key]
    m = npp_multiplier(f, current_date)
    f.cache[key] = m
    return m
end

# -------------------------------------------------------------------
# ASC-only bootstrap: build the initial MarineEnvironment + habitat capacities
# entirely from the .asc XML (no NetCDF). Produces arrays with exactly the shapes
# the rest of the model expects:
#   envi.data["temp"] :: (lonres, latres, depthres, 12)   surface temp over depth
#   capacities        :: (lonres, latres, 12, n_spec+n_resource)
# Both are 12-month climatologies seeded from the start year's monthly drivers;
# from the first timestep onward update_environment_from_asc! overwrites the
# current calendar-month slice with the true absolute-month data, so this is just
# a valid initial condition for placement / resource init.
# -------------------------------------------------------------------
function bootstrap_environment_from_asc(f::EnvForcing, files, grid, arch,
                                        start_date::Date, n_spec::Integer, n_resource::Integer,
                                        plt_diags; var_alias::Dict{String,String}=default_var_alias())
    g(name) = grid[grid.Name .== name, :Value][1]
    lonres = Int(g("lonres")); latres = Int(g("latres")); depthres = Int(g("depthres"))
    boot_year = year(start_date)

    prefs_df = CSV.read(files[files.File .== "envi_pref", :Destination][1], DataFrame)
    trait    = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait", :Destination][1], DataFrame))))
    resource = Dict(pairs(eachcol(CSV.read(files[files.File .== "resource_trait", :Destination][1], DataFrame))))
    spec_names = String.(vcat(trait[:SpeciesLong][1:n_spec], resource[:SpeciesLong][1:n_resource]))

    # --- temperature field (lon,lat,depth,12): surface temp broadcast over depth -
    temp_cpu = fill(NaN32, lonres, latres, depthres, 12)
    for m in 1:12
        tos = env_layer(f, "tos", Date(boot_year, m, 1))
        tos === nothing && continue
        fillv = Float32(_nanmean(tos))
        @inbounds for z in 1:depthres, lat in 1:latres, lon in 1:lonres
            v = tos[lon, lat]
            temp_cpu[lon, lat, z, m] = isfinite(v) ? v : fillv
        end
    end
    envi = MarineEnvironment(Dict{String,AbstractArray}("temp" => array_type(arch)(temp_cpu)), 1)

    # --- optional static bathymetry layer (lon,lat) -----------------------------
    # If the env XML declares a `bathymetry` variable (a single static .asc, e.g.
    # dropped directly in EnvData), load it here so initial placement and any
    # future depth-aware logic can use envi.data["bathymetry"]. Missing -> skipped
    # (placement falls back to capacity>0, which already excludes land).
    if haskey(f.vars, "bathymetry")
        bath = env_layer(f, "bathymetry", start_date)
        if bath !== nothing
            envi.data["bathymetry"] = array_type(arch)(bath)
            @info "Loaded static bathymetry layer $(size(bath)) into environment."
        end
    end

    # --- habitat capacity (lon,lat,12,nspec_tot) from the driver envelope --------
    capacities_cpu = ones(Float32, lonres, latres, 12, n_spec + n_resource)
    for m in 1:12
        update_capacity_from_drivers!(capacities_cpu, f, Date(boot_year, m, 1),
                                      prefs_df, spec_names, var_alias)
    end

    if plt_diags == 1
        try
            res_dir = files[files.File .== "res_dir", :Destination][1]
            outdir = joinpath(res_dir, "diags", "Capacities")
            isdir(outdir) && rm(outdir, recursive=true); mkpath(outdir)
            for i in 1:length(spec_names), m in 1:12
                p = heatmap(capacities_cpu[:, :, m, i]', title="", xlabel="Lon idx",
                            ylabel="Lat idx", c=:viridis, clims=(0, 1))
                savefig(p, joinpath(outdir, "$(spec_names[i])_month_$(m)_capacity.png"))
            end
            @info "Bootstrap habitat-capacity maps exported to $(abspath(outdir))"
        catch err
            @warn "Could not export bootstrap capacity diagnostics: $err"
        end
    end

    @info "Bootstrapped environment from ASC for $boot_year: temp $(size(temp_cpu)), capacities $(size(capacities_cpu))."
    return envi, array_type(arch)(capacities_cpu)
end
