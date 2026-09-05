# The v35 solution-record block exactly as PowerFlows wrote it before the block carried
# any solve information. An export with no solve attached must still produce this, byte
# for byte, or every existing v35 export changes.
const SOLUTION_RECORDS_GOLDEN_DEFAULT = """GENERAL, THRSHZ=0.0001, PQBRAK=0.7, BLOWUP=5.0, MaxIsolLvls=4, CAMaxReptSln=20, ChkDupCntLbl=0
GAUSS, ITMX=100, ACCP=1.6, ACCQ=1.6, ACCM=1.0, TOL=0.0001
NEWTON, ITMXN=20, ACCN=1.0, TOLN=0.1, VCTOLQ=0.1, VCTOLV=0.00001, DVLIM=0.99, NDVFCT=0.99
ADJUST, ADJTHR=0.005, ACCTAP=1.0, TAPLIM=0.05, SWVBND=100.0, MXTPSS=99, MXSWIM=10
TYSL, ITMXTY=20, ACCTY=1.0, TOLTY=0.00001
SOLVER,     , ACTAPS=0, AREAIN=0, PHSHFT=0, DCTAPS=0, SWSHNT=0, FLATST=0, VARLIM=0, NONDIV=0
RATING, 1, "RATE1 ", "RATING SET 1                    "
RATING, 2, "RATE2 ", "RATING SET 2                    "
RATING, 3, "RATE3 ", "RATING SET 3                    "
RATING, 4, "RATE4 ", "RATING SET 4                    "
RATING, 5, "RATE5 ", "RATING SET 5                    "
RATING, 6, "RATE6 ", "RATING SET 6                    "
RATING, 7, "RATE7 ", "RATING SET 7                    "
RATING, 8, "RATE8 ", "RATING SET 8                    "
RATING, 9, "RATE9 ", "RATING SET 9                    "
RATING,10, "RATE10", "RATING SET 10                   "
RATING,11, "RATE11", "RATING SET 11                   "
RATING,12, "RATE12", "RATING SET 12                   "
"""

_write_records(pf, params; base_power = 100.0) =
    sprint(io -> PF.write_solution_records(io, pf, params, base_power))

@testset "No solve attached writes the format defaults byte for byte" begin
    @test _write_records(nothing, nothing) == SOLUTION_RECORDS_GOLDEN_DEFAULT
end

@testset "Decimal rendering avoids scientific notation" begin
    # `string(1e-5)` is "1.0e-5", which would silently reformat records the writer
    # otherwise passes through unchanged.
    @test PF._decimal_string(1e-5) == "0.00001"
    @test PF._decimal_string(1e-7) == "0.0000001"
    @test PF._decimal_string(1.5e-5) == "0.000015"
    @test PF._decimal_string(0.1) == "0.1"
    @test PF._decimal_string(5.0) == "5.0"
    @test PF._decimal_string(100.0) == "100.0"
    @test PF._decimal_string(12) == "12"
end

@testset "A solve maps onto the solution records" begin
    params = SolutionParameters(;
        tol = 1e-6,
        maxIterations = 30,
        check_reactive_power_limits = true,
        control_discrete_devices = true,
        area_interchange_control = true,
        enhanced_flat_start = true,
    )
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; solution_parameters = params)
    v = PF.solution_record_values(pf, PF.get_solution_parameters(pf), 100.0)

    @test v.solver == PF.SOLUTION_RECORD_SOLVER_FULL_NEWTON
    @test v.itmxn == 30
    # PowerFlows works in per unit; the record is a MW/MVAr mismatch.
    @test v.toln ≈ 1e-4
    @test v.actaps == 1
    @test v.swshnt == 1
    @test v.areain == 1
    @test v.varlim == 0     # limits applied
    @test v.flatst == 1
    # No PowerFlows counterpart exists for either.
    @test v.phshft == 0
    @test v.dctaps == 0

    text = _write_records(pf, PF.get_solution_parameters(pf))
    @test occursin("ACTAPS=1", text)
    @test occursin("AREAIN=1", text)
    @test occursin("SWSHNT=1", text)
    @test occursin("TOLN=0.0001", text)
    @test occursin("ITMXN=30", text)
    @test occursin("FNSL", text)
end

@testset "Reactive limits and the iteration budget follow the solver" begin
    off = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(; check_reactive_power_limits = false),
    )
    # -1 is the format's "ignore the limits".
    @test PF.solution_record_values(off, PF.get_solution_parameters(off), 100.0).varlim ==
          -1
    # An unset cap reports the solver's own default rather than the format's.
    @test PF.solution_record_values(off, PF.get_solution_parameters(off), 100.0).itmxn ==
          PF.DEFAULT_NR_MAX_ITER

    fd = ACPolarPowerFlow{FastDecoupledACPowerFlow}()
    v = PF.solution_record_values(fd, PF.get_solution_parameters(fd), 100.0)
    @test v.solver == PF.SOLUTION_RECORD_SOLVER_DECOUPLED
    @test v.itmxn == PF.DEFAULT_FD_MAX_ITER
    @test v.nondiv == 1  # non-divergent backtracking is the fast-decoupled default
    @test v.dvlim == PF.DEFAULT_FD_DVLIM
end

@testset "Reading solution records back" begin
    dir = mktempdir()
    path = joinpath(dir, "case.raw")
    open(path, "w") do io
        println(io, "0, 100.0, 35, 0, 1, 60.0")
        println(io, "a case")
        println(io)
        print(io, SOLUTION_RECORDS_GOLDEN_DEFAULT)
        println(io, "0 / END OF SYSTEM-WIDE DATA, BEGIN BUS DATA")
        println(io, "1, 'BUS 1', 230.0, 3, 1, 1, 1, 1.0, 0.0")
    end

    v = PF.read_solution_records(path)
    @test !isnothing(v)
    @test v.itmxn == 20
    @test v.toln == 0.1
    @test v.solver == ""
    @test v.varlim == 0
    @test v.areain == 0

    params = read_solution_parameters(path)
    @test params.tol ≈ 0.1 / 100.0
    @test params.maxIterations == 20
    @test params.check_reactive_power_limits    # VARLIM=0 applies the limits
    @test !params.control_discrete_devices
    @test !params.area_interchange_control

    # A v33 file carries no solution records at all.
    @test isnothing(read_solution_parameters(joinpath(TEST_DATA_DIR, "14_bus.raw")))

    # A v35 file whose block is absent is not read as bus data.
    bare = joinpath(dir, "bare.raw")
    open(bare, "w") do io
        println(io, "0, 100.0, 35, 0, 1, 60.0")
        println(io, "a case")
        println(io, "1, 'BUS 1', 230.0, 3, 1, 1, 1, 1.0, 0.0")
    end
    @test isnothing(read_solution_parameters(bare))
end

@testset "Solution records round trip through the file" begin
    params = SolutionParameters(;
        tol = 1e-6,
        maxIterations = 30,
        check_reactive_power_limits = true,
        control_discrete_devices = true,
        area_interchange_control = true,
    )
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; solution_parameters = params)

    dir = mktempdir()
    path = joinpath(dir, "case.raw")
    open(path, "w") do io
        println(io, "0, 100.0, 35, 0, 1, 60.0")
        println(io, "a case")
        println(io)
        PF.write_solution_records(io, pf, params, 100.0)
        println(io, "0 / END OF SYSTEM-WIDE DATA, BEGIN BUS DATA")
    end

    recovered = read_solution_parameters(path)
    @test recovered.tol ≈ params.tol
    @test recovered.maxIterations == params.maxIterations
    @test recovered.check_reactive_power_limits
    @test recovered.control_discrete_devices
    @test recovered.area_interchange_control
    @test recovered.tie_definition === :lines_only

    # Parse, then write: the block must come back identical.
    written = _write_records(pf, params)
    reparsed = PF.read_solution_records(path)
    @test sprint(io -> PF.write_solution_records(io, nothing, nothing, 100.0)) ==
          SOLUTION_RECORDS_GOLDEN_DEFAULT
    @test reparsed == PF.solution_record_values(pf, params, 100.0)
    @test occursin("ITMXN=30", written)
end

@testset "Unrecognized records and quoted labels survive the reader" begin
    dir = mktempdir()
    path = joinpath(dir, "case.raw")
    open(path, "w") do io
        println(io, "0, 100.0, 35, 0, 1, 60.0")
        println(io, "a case")
        println(io, "@! a comment line")
        println(io, "FUTURERECORD, SOMETHING=1")
        println(io, "NEWTON, ITMXN=42, TOLN=0.5, UNKNOWNFIELD=9")
        println(io, "RATING, 1, \"RATE1 \", \"A LABEL, WITH A COMMA\"")
        println(io, "0 / END")
    end
    v = PF.read_solution_records(path)
    @test v.itmxn == 42
    @test v.toln == 0.5
end
