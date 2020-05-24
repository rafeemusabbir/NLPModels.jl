mutable struct LinearRegression
  x :: Vector
  y :: Vector
end

function (regr::LinearRegression)(beta)
  r = regr.y .- beta[1] - beta[2] * regr.x
  return dot(r, r) / 2
end

function test_autodiff_model()
  x0 = zeros(2)
  f(x) = dot(x,x)
  nlp = ADNLPModel(f, x0)

  c(x) = [sum(x) - 1]
  nlp = ADNLPModel(f, x0, c, [0], [0])
  @test obj(nlp, x0) == f(x0)

  x = range(-1, stop=1, length=100)
  y = 2x .+ 3 + randn(100) * 0.1
  regr = LinearRegression(x, y)
  nlp = ADNLPModel(regr, ones(2))
  β = [ones(100) x] \ y
  @test abs(obj(nlp, β) - norm(y .- β[1] - β[2] * x)^2 / 2) < 1e-12
  @test norm(grad(nlp, β)) < 1e-12

  @testset "Constructors for ADNLPModel" begin
    nlp = ADNLPModel(f, x0)
    nlp = ADNLPModel(f, x0, nvar=2)
    nlp = ADNLPModel(f, x0, nvar=2, lvar=zeros(2))
    nlp = ADNLPModel(f, x0, nvar=2, uvar=zeros(2))
    nlp = ADNLPModel(f, x0, nvar=2, lvar=zeros(2), uvar=zeros(2))
    @test_throws DimensionError ADNLPModel(f, x0, nvar=2, lvar=zeros(1))
    @test_throws DimensionError ADNLPModel(f, x0, nvar=2, uvar=zeros(1))
    nlp = ADNLPModel(f, x0, c=c, ncon=1)
    nlp = ADNLPModel(f, x0, c=c, ncon=1, lcon=[0.0])
    nlp = ADNLPModel(f, x0, c=c, ncon=1, ucon=[0.0])
    nlp = ADNLPModel(f, x0, c=c, ncon=1, lcon=[0.0], ucon=[0.0])
    nlp = ADNLPModel(f, x0, c=c, ncon=1, lcon=[0.0], ucon=[0.0], y0=[1.0])
    @test_throws DimensionError ADNLPModel(f, x0, c=c, ncon=1, lcon=zeros(2))
    @test_throws DimensionError ADNLPModel(f, x0, c=c, ncon=1, ucon=zeros(2))
    @test_throws DimensionError ADNLPModel(f, x0, c=c, ncon=1, y0=zeros(2))
    nlp = ADNLPModel(f, x0, -ones(2), ones(2))
    nlp = ADNLPModel(f, x0, c, [-1.0], [1.0])
    nlp = ADNLPModel(f, x0, -ones(2), ones(2), c, [-1.0], [1.0])
  end
end

test_autodiff_model()
