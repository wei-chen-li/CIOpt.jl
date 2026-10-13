"""Build the Julia AuLa subproblem from a MPCCModels model."""
buildMPCC(model::MPCCModels.AbstractMPCCModel, config::AulaConfig=AulaConfig()) =
    AulaSubproblem(model, validate(config))
