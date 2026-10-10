"""
Check if a device has attribute 'active_power' for active power consumption or generation.
"""
function contributes_active_power(::T) where {T <: PSY.Device}
    throw(
        IS.NotImplementedError(
            "contributes_active_power not implemented for this device type $(T)",
        ),
    )
end

IS.@scoped_enum(PowerContributionType, INJECTION = 1, WITHDRAWAL = 2)
function active_power_contribution_type(::T) where {T <: PSY.Device}
    throw(
        IS.NotImplementedError(
            "active_power_contribution_type not implemented for this device type $(T)",
        ),
    )
end

# most static components inject power
contributes_active_power(::PSY.StaticInjection) = true
active_power_contribution_type(::PSY.StaticInjection) = PowerContributionType.INJECTION
# carve out the exceptions: we handle constant impedance/current loads separately
contributes_active_power(::Union{PSY.FixedAdmittance, PSY.SwitchedAdmittance}) = false
# the getter is named differently, `get_constant_active_power`, so handle separately
contributes_active_power(::Union{PSY.StandardLoad, PSY.InterruptibleStandardLoad}) = false
# withdraws active power, but getter is named differently: `get_active_power_losses`
contributes_active_power(::PSY.SynchronousCondenser) = false
# voltage dependent: `_get_withdrawals!` maps it onto the ZIP terms by exponent
contributes_active_power(::PSY.ExponentialLoad) = false
# not fully supported yet.
contributes_active_power(::PSY.FACTSControlDevice) = false
# loads withdraw power.
active_power_contribution_type(::PSY.ElectricLoad) = PowerContributionType.WITHDRAWAL

function contributes_reactive_power(::T) where {T <: PSY.Device}
    throw(
        IS.NotImplementedError(
            "contributes_reactive_power not implemented for this device type $(T)",
        ),
    )
end

function reactive_power_contribution_type(::T) where {T <: PSY.Device}
    throw(
        IS.NotImplementedError(
            "reactive_power_contribution_sign not implemented for this device type $(T)",
        ),
    )
end

# most static components contribute reactive power
contributes_reactive_power(::PSY.StaticInjection) = true
reactive_power_contribution_type(::PSY.StaticInjection) = PowerContributionType.INJECTION
# handle constant impedance loads separately
contributes_reactive_power(::Union{PSY.FixedAdmittance, PSY.SwitchedAdmittance}) = false
# the getter is named differently, `get_constant_reactive_power`, so handle separately
contributes_reactive_power(::Union{PSY.StandardLoad, PSY.InterruptibleStandardLoad}) = false
# interconnecting converters do not support reactive power
contributes_reactive_power(::PSY.InterconnectingConverter) = false
contributes_reactive_power(::PSY.ExponentialLoad) = false
# not fully supported yet.
contributes_reactive_power(::PSY.FACTSControlDevice) = false
# loads withdraw reactive power
reactive_power_contribution_type(::PSY.ElectricLoad) = PowerContributionType.WITHDRAWAL
