using ForwardDiff

export ADNLPModel

mutable struct ADNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f
  c
end

show_header(io :: IO, nlp :: ADNLPModel) = println(io, "ADNLPModel - Model with automatic differentiation")

"""ADNLPModel is an AbstractNLPModel using ForwardDiff to compute the
derivatives.
In this interface, the objective function ``f`` and an initial estimate are
required. If there are constraints, the function
``c:ℝⁿ → ℝᵐ``  and the vectors
``c_L`` and ``c_U`` also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.

````
ADNLPModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0 = zeros,
  c = x->[], lcon = [], ucon = [], name = "Generic")
````

  - `f` - The objective function ``f``. Should be callable;
  - `x0 :: AbstractVector` - The initial point of the problem;
  - `lvar :: AbstractVector` - ``ℓ``, the lower bound of the variables;
  - `uvar :: AbstractVector` - ``u``, the upper bound of the variables;
  - `c` - The constraints function ``c``. Should be callable;
  - `y0 :: AbstractVector` - The initial value of the Lagrangian estimates;
  - `lcon :: AbstractVector` - ``c_L``, the lower bounds of the constraints function;
  - `ucon :: AbstractVector` - ``c_U``, the upper bounds of the constraints function;
  - `name :: String` - A name for the model.

The functions follow the same restrictions of ForwardDiff functions, summarised
here:

  - The function can only be composed of generic Julia functions;
  - The function must accept only one argument;
  - The function's argument must accept a subtype of AbstractVector;
  - The function should be type-stable.

For contrained problems, the function ``c`` is required, and it must return
an array even when m = 1,
and ``c_L`` and ``c_U`` should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of ``c_L`` and ``c_U`` should be the
same.
"""
function ADNLPModel(f, x0::AbstractVector{T}; nvar :: Integer = length(x0), ncon :: Integer = 0,
                    lvar::AbstractVector = fill(T(-Inf), nvar), uvar::AbstractVector = fill(T(Inf), nvar),
                    lcon::AbstractVector = fill(T(-Inf), ncon), ucon::AbstractVector = fill(T(Inf), ncon),
                    c = (args...)->T[], y0::AbstractVector = zeros(T, ncon),
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[]) where T

  @lencheck nvar x0 lvar uvar
  @lencheck ncon y0 lcon ucon

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, c)
end

"""
    nlp = ADNLPModel(f, x0, lvar, uvar; kwargs...)
"""
function ADNLPModel(f, x0::AbstractVector, lvar::AbstractVector, uvar::AbstractVector; kwargs...)
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar

  return ADNLPModel(f, x0, nvar=nvar, lvar=lvar, uvar=uvar; kwargs...)
end

"""
    nlp = ADNLPModel(f, x0, c, lcon, ucon; kwargs...)
"""
function ADNLPModel(f, x0::AbstractVector, c, lcon::AbstractVector, ucon::AbstractVector; kwargs...)
  ncon = length(lcon)
  @lencheck ncon ucon

  return ADNLPModel(f, x0, nvar=length(x0), ncon=ncon, c=c, lcon=lcon, ucon=ucon; kwargs...)
end

"""
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon; kwargs...)
"""
function ADNLPModel(f, x0::AbstractVector, lvar::AbstractVector, uvar::AbstractVector,
                    c, lcon::AbstractVector, ucon::AbstractVector; kwargs...)
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar
  ncon = length(lcon)
  @lencheck ncon ucon

  return ADNLPModel(f, x0, nvar=nvar, lvar=lvar, uvar=uvar, ncon=ncon, c=c, lcon=lcon, ucon=ucon; kwargs...)
end

function obj(nlp :: ADNLPModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function grad!(nlp :: ADNLPModel, x :: AbstractVector, g :: AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  ForwardDiff.gradient!(g, nlp.f, x)
  return g
end

function cons!(nlp :: ADNLPModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  c .= nlp.c(x)
  return c
end

function jac(nlp :: ADNLPModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)
  return ForwardDiff.jacobian(nlp.c, x)
end

function jac_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzj rows cols
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord!(nlp :: ADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  Jx = ForwardDiff.jacobian(nlp.c, x)
  vals .= Jx[:]
  return vals
end

function jprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  Jv .= ForwardDiff.derivative(t -> nlp.c(x + t * v), 0)
  return Jv
end

function jtprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  Jtv .= ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
  return Jtv
end

function hess(nlp :: ADNLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess(nlp :: ADNLPModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: ADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp :: ADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  n = nlp.meta.nvar
  @lencheck n x v Hv
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nlp :: ADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  n = nlp.meta.nvar
  @lencheck n x v Hv
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
