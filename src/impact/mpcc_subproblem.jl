"""Build the Julia AuLa subproblem from a MPCCModels model."""
buildMPCC(model::MPCCModels.AbstractMPCCModel, config::AulaConfig=AulaConfig()) =
    AulaSubproblem(model, validate(config))

"""Build from a model already guaranteed to use MPCCModels vertical form."""
function build_vertical_mpcc(model::MPCCModels.AbstractMPCCModel,
                             config::AulaConfig=AulaConfig())
    MPCCModels.is_vertical(model) || throw(ArgumentError("expected a vertical-form MPCC"))
    return AulaSubproblem(model, validate(config))
end
