# ===================================================================
# Core Data Structures
# ===================================================================

# Gated progress printing (verbose flag in RUNTIME_CONFIG; safe if undefined yet).
function vprint(args...)
    (isdefined(@__MODULE__, :RUNTIME_CONFIG) && get(RUNTIME_CONFIG, :verbose, 1) != 0) && print(args...)
end
function vprintln(args...)
    (isdefined(@__MODULE__, :RUNTIME_CONFIG) && get(RUNTIME_CONFIG, :verbose, 1) != 0) && println(args...)
end

# Cached CSV reader: input files don't change during a run, so parse once and
# reuse. Eliminates repeated disk reads in hot loops with no change in results.
const _CSV_CACHE = Dict{String, Any}()
function cached_csv(path::AbstractString)
    key = abspath(String(path))
    get!(_CSV_CACHE, key) do
        CSV.read(key, DataFrame)
    end
end
clear_csv_cache!() = (empty!(_CSV_CACHE); nothing)
cached_csv_for(files, label::AbstractString) =
    cached_csv(files[files.File .== label, :Destination][1])

# Struct for a single species' agents and parameters
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

# Container for all focal species
struct individuals
    animals::NamedTuple
end

# The `resource` struct is no longer needed for the grid-based system.
# mutable struct resource ...

# Temporary container for prey information during search
mutable struct PreyInfo
    Predator::Int
    Sp::Int
    Ind::Int
    Type::Int
    Length::Float32
    Biomass::Float32
    Distance::Float32
end

# Defines the gear selectivity for a species within a fishery
struct Selectivity
    species::String
    sel_type::Int8 # 1=logistic, 2=knife_edge, 3=dome_shaped
    p1::Float32    # L50 for logistic/knife_edge, L50_1 for dome
    p2::Float32    # Slope for logistic, Slope_1 for dome
    p3::Float32    # L50_2 for dome
    p4::Float32    # Slope_2 for dome
end

# Defines a single fishery's regulations and properties
mutable struct Fishery
    name::String
    target_species::Vector{String}
    bycatch_species::Vector{String}
    selectivities::Dict{String, Selectivity}
    quota::Float32
    cumulative_catch::Float32
    cumulative_inds::Int32
    # --- Quota rule (dynamic harvest control) ---
    # quota_mode: 1 = fixed annual quota (quota held constant, as before);
    #             2 = biomass-linked rate, recomputed each year on Jan 1 as
    #                 quota = harvest_rate * (current target-species biomass, mt),
    #                 so the fishery grows/shrinks with the stock.
    quota_mode::Int32
    harvest_rate::Float32      # annual exploitation fraction of target biomass (mode 2)
    quota_min::Float32         # lower cap on the annual quota (mt); 0 = none
    quota_max::Float32         # upper cap on the annual quota (mt); <=0 = none
    b_lim::Float32             # biomass (mt) at/below which the quota ramps to 0 (HCR); 0 = no ramp
    b_thr::Float32             # biomass (mt) at/above which the full rate applies (HCR); 0 = no ramp
    season::Tuple{Int, Int}
    area::Tuple{Tuple{Float32, Float32}, Tuple{Float32, Float32}, Tuple{Float32, Float32}}
    slot_limit::Tuple{Float32, Float32}
    bag_limit::Int32
    effort_days::Int
    mean_length_catch::Float64
    mean_weight_catch::Float64
    bycatch_tonnage::Float64
    bycatch_inds::Int
    quota_base::Float32        # baseline annual quota (mt) from CSV; anchor for scaled mode
    b_ref::Float32             # reference biomass (mt); 0 = capture current biomass on first update
end

# A flexible struct to hold any environmental data loaded from a NetCDF file
mutable struct MarineEnvironment
    data::Dict{String, AbstractArray}
    ts::Int # To hold the current month index
end

# Holds vertical distribution data for agents
mutable struct MarineDepths
    focal_day::DataFrame
    focal_night::DataFrame 
    resource_day::DataFrame
    resource_night::DataFrame
    grid::DataFrame
end

# The main model struct, containing the entire state of the simulation
mutable struct MarineModel
    arch::Architecture
    environment::MarineEnvironment
    depths::MarineDepths
    fishing::Vector{Fishery}
    t::Float32
    iteration::Int32
    dt::Float32
    individuals::individuals
    resources::NamedTuple{(:biomass, :capacity), Tuple{AbstractArray{Float32, 4}, AbstractArray{Float32, 4}}}
    resource_trait::DataFrame
    capacities::AbstractArray
    ninds::Int64
    n_species::Int32
    n_resource::Int32
    abund::Vector{Int64}
    bioms::Vector{Float32}
    init_abund::Vector{Int64}
    files::DataFrame
    output_dt::Int32
    spinup::Int32
    foraging_attempts::Int32
    plt_diags::Int32
    size_bin_thresholds::AbstractMatrix{Float32}
    daily_birth_counters::Vector{Int}
end

# ===================================================================
# Helper and Utility Functions
# ===================================================================

# --- Spatial Indexing Helpers ---
@inline function get_cell_id(x, y, z, lonres, latres)
    return x + (y - 1) * lonres + (z - 1) * lonres * latres
end

@inline function get_cell_xyz(id, lonres, latres)
    z = div(id - 1, lonres * latres) + 1
    rem_id = (id - 1) % (lonres * latres)
    y = div(rem_id, lonres) + 1
    x = rem_id % lonres + 1
    return x, y, z
end

# --- Statistical and Distribution Helpers ---
function gaussmix(n, m1, m2, m3, s1, s2, s3, l1, l2)
    I = rand(n) .< l1
    I2 = rand(n) .< l1 .+ l2
    z = [rand(I[i] ? Normal(m1, s1) : (I2[i] ? Normal(m2, s2) : Normal(m3, s3))) for i in 1:n]
    return z
end

function multimodal_distribution(x, means, stds, weights)
    pdf_values = [weights[i] * pdf(Normal(means[i], stds[i]), x) for i in 1:length(means)]
    return sum(pdf_values)
end

function get_target_z(sp, dist)
    return gaussmix(1, dist[sp, "mu1"], dist[sp, "mu2"], dist[sp, "mu3"], dist[sp, "sigma1"], dist[sp, "sigma2"], dist[sp, "sigma3"], dist[sp, "lambda1"], dist[sp, "lambda2"])[1]
end

function lognormal_params_from_maxsize(max_size::Real)
    median_val = 1/3 * max_size
    percentile = 0.95
    μ = log(median_val)
    z_score = quantile(Normal(0, 1), percentile)
    σ = (log(max_size) - μ) / z_score
    return μ, σ
end

function lognormal_params_from_minmax(min_size::Real, max_size::Real)
    # Ensure min_size is positive to avoid log(0) errors
    min_size = max(min_size, 1e-6)
    
    # Ensure max_size is always greater than min_size
    if min_size >= max_size
        max_size = min_size * 1.1
    end

    log_min, log_max = log(min_size), log(max_size)
    
    # Use the z-score for the 97.5th percentile to cover 95% of the distribution
    z = 1.96 
    
    # The mean of the log-transformed values
    μ = (log_max + log_min) / 2.0
    
    # The standard deviation of the log-transformed values
    σ = (log_max - log_min) / (2.0 * z)
    
    return μ, σ
end

# --- Geometric and Distance Helpers ---
function haversine(lat1, lon1, lat2, lon2)
    R = 6371000.0 # Earth radius in meters
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    c = 2 * atan(sqrt(a), sqrt(1 - a))
    return R * c
end

# --- Data Loading and Initialization Helpers ---
function generate_depths(files)
    focal_day = CSV.read(files[files.File .== "focal_z_dist_day", :Destination][1], DataFrame)
    focal_night = CSV.read(files[files.File .== "focal_z_dist_night", :Destination][1], DataFrame)
    resource_day = CSV.read(files[files.File .== "resource_z_dist_day", :Destination][1], DataFrame)
    resource_night = CSV.read(files[files.File .== "resource_z_dist_night", :Destination][1], DataFrame)
    grid = CSV.read(files[files.File .== "grid",:Destination][1], DataFrame)
    return MarineDepths(focal_day, focal_night, resource_day, resource_night, grid)
end

function load_ascii_raster(file_path::String)
    open(file_path, "r") do f
        header = Dict{String, Float32}()
        for _ in 1:6
            line = readline(f)
            key, val = split(line)
            header[key] = parse(Float32, val)
        end
        data = readdlm(f)
        return data
    end
end

# --- Physical Model Helpers ---
function ipar_curve(time, peak_ipar=450, peak_time=12, width=4)
    adj_time = time/60
    return peak_ipar * exp(-((adj_time - peak_time)^2) / (2 * width^2))
end

function generate_random_directions(n::Int)
    θ = 2π .* rand(Float32, n)
    φ = acos.(2f0 .* rand(Float32, n) .- 1f0)
    dx = sin.(φ) .* cos.(θ)
    dy = sin.(φ) .* sin.(θ)
    dz = cos.(φ)
    return dx, dy, dz
end

function resize_agent_storage!(model::MarineModel, sp::Int, new_maxN::Int)
    arch = model.arch
    current_data = model.individuals.animals[sp].data
    current_maxN = length(current_data.x)

    # Ensure we are actually growing the arrays
    if new_maxN <= current_maxN
        @warn "Resize called with new_maxN ($new_maxN) <= current_maxN ($current_maxN). No action taken."
        return
    end

    @info "Resizing agent storage for species $sp: $current_maxN -> $new_maxN"

    # --- 1. Create a new, larger StructArray on the target architecture ---
    fields = propertynames(current_data)
    
    # Create new empty arrays with the new size
    new_arrays = Tuple(similar(getproperty(current_data, f), new_maxN) for f in fields)
    
    # Construct the new StructArray
    new_device_data = StructArray{eltype(current_data)}(new_arrays)

    # --- 2. Copy old data and initialize new slots ---
    for field in fields
        current_array = getproperty(current_data, field)
        new_array = getproperty(new_device_data, field)
        
        # A. Copy the existing data to the start of the new array
        copyto!(@view(new_array[1:current_maxN]), current_array)
        
        # B. *** FIX: Initialize the new, empty slots to zero ***
        # This prevents reading garbage data from uninitialized memory.
        # The `eltype` ensures we fill with the correct type of zero (e.g., 0.0f0, 0).
        fill!(@view(new_array[current_maxN+1:end]), zero(eltype(new_array)))
    end
    
    # --- 3. Replace the old data field in the model ---
    model.individuals.animals[sp].data = new_device_data
    
    return nothing
end

"""
    create_size_bin_matrix(agent_traits, resource_traits, n_bins, arch)

Creates a matrix of size bin thresholds. Each column corresponds to a species,
and the thresholds are calculated by dividing the species' size range
(Min_Size to Max_Size) into 'n_bins' logarithmically-spaced intervals.
"""
function create_size_bin_matrix(agent_traits, resource_traits, n_bins::Int32, arch)
    # Combine all species into one list for processing
    all_species_traits = vcat(DataFrame(agent_traits), resource_traits, cols=:union)
    n_total_species = nrow(all_species_traits)
    
    # The number of internal thresholds is one less than the number of bins
    n_thresholds = n_bins - 1
    
    # --- CHANGE: Create a matrix to hold min, max, and all thresholds ---
    # The total number of rows will be n_thresholds + 2 (for min and max)
    # which is equivalent to n_bins + 1
    thresholds_cpu = zeros(Float32, n_bins + 1, n_total_species)

    for i in 1:n_total_species
        min_s = all_species_traits.Min_Size[i]
        max_s = all_species_traits.Max_Size[i]
        
        # Avoid errors with log(0) if a species has a min size of 0
        if min_s <= 0; min_s = 0.1f0; end

        # --- CHANGE: Populate the first and last rows with Min/Max Size ---
        thresholds_cpu[1, i] = min_s
        thresholds_cpu[end, i] = max_s
        # --- END CHANGE ---

        log_min = log(min_s)
        log_max = log(max_s)
        log_step = (log_max - log_min) / n_bins

        # --- CHANGE: Populate the intermediate rows with the thresholds ---
        for j in 1:n_thresholds
            # The thresholds are placed in rows 2 through n_thresholds + 1
            thresholds_cpu[j + 1, i] = exp(log_min + j * log_step)
        end
        # --- END CHANGE ---
    end

    # Return the matrix uploaded to the target device (e.g., GPU)
    return array_type(arch)(thresholds_cpu)
end

"""
    find_species_size_bin(value, species_idx, bins_matrix)

An @inline device function to be used inside a GPU kernel. It determines which
size bin a given `value` (e.g., an agent's length) falls into for a specific
species by looking up the correct column in the `bins_matrix`.
"""
@inline function find_species_size_bin(value, species_idx, bins_matrix)
    # The number of bins is the number of rows in the matrix minus one.
    n_bins = size(bins_matrix, 1) - 1

    # The actual bin thresholds start at the second row of the matrix.
    # We loop from the first threshold (row 2) up to the last one (row `n_bins`).
    for i in 2:n_bins
        # Check if the value is less than the upper threshold of the current bin.
        # The threshold for bin `i-1` is at row `i`.
        if value < bins_matrix[i, species_idx]
            # Example: If value is less than the threshold at row 2, it's in bin 1.
            return i - 1
        end
    end
    
    # If the value is greater than or equal to the last threshold (at row `n_bins`),
    # it belongs in the last bin (`n_bins`).
    return n_bins
end

@inline function custom_erf(x::Float32)
    p  = 0.3275911f0
    a1 = 0.254829592f0; a2 = -0.284496736f0; a3 = 1.421413741f0
    a4 = -1.453152027f0; a5 = 1.061405429f0
    sign = ifelse(x >= 0.0f0, 1.0f0, -1.0f0)
    x_abs = abs(x)
    t = 1.0f0 / (1.0f0 + p * x_abs)
    y = 1.0f0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * exp(-x_abs * x_abs)
    return sign * y
end

@inline function calculate_proportion_in_bin(lower_bound::Float32, upper_bound::Float32, μ::Float32, σ::Float32)
        sqrt2 = 1.41421356237f0

    if lower_bound < 1.0f-20 
        # If the lower bound is effectively zero, calculate CDF at the upper bound
        z_upper = (log(upper_bound) - μ) / σ
        return 0.5f0 * (1.0f0 + custom_erf(z_upper / sqrt2))
    else
        # Otherwise, calculate the difference between the two CDFs
        z_lower = (log(lower_bound) - μ) / σ
        z_upper = (log(upper_bound) - μ) / σ
        cdf_upper = 0.5f0 * (1.0f0 + custom_erf(z_upper / sqrt2))
        cdf_lower = 0.5f0 * (1.0f0 + custom_erf(z_lower / sqrt2))
        return cdf_upper - cdf_lower
    end
end