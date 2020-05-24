function autodiff_nls_test()
  @testset "autodiff_nls_test" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2]
    nls = ADNLSModel(F, 2, 2)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol=1e-8)
  end

  @testset "Constructors for ADNLSModel" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2; x[1] * x[2]]
    x0 = ones(2)
    c(x) = [sum(x) - 1]
    nlp = ADNLSModel(F, x0, 3)
    nlp = ADNLSModel(F, x0, 3, nvar=2)
    nlp = ADNLSModel(F, x0, 3, nvar=2, lvar=zeros(2))
    nlp = ADNLSModel(F, x0, 3, nvar=2, uvar=zeros(2))
    nlp = ADNLSModel(F, x0, 3, nvar=2, lvar=zeros(2), uvar=zeros(2))
    @test_throws DimensionError ADNLSModel(F, x0, 3, nvar=2, lvar=zeros(1))
    @test_throws DimensionError ADNLSModel(F, x0, 3, nvar=2, uvar=zeros(1))
    nlp = ADNLSModel(F, x0, 3, c=c, ncon=1)
    nlp = ADNLSModel(F, x0, 3, c=c, ncon=1, lcon=[0.0])
    nlp = ADNLSModel(F, x0, 3, c=c, ncon=1, ucon=[0.0])
    nlp = ADNLSModel(F, x0, 3, c=c, ncon=1, lcon=[0.0], ucon=[0.0])
    nlp = ADNLSModel(F, x0, 3, c=c, ncon=1, lcon=[0.0], ucon=[0.0], y0=[1.0])
    @test_throws DimensionError ADNLSModel(F, x0, 3, c=c, ncon=1, lcon=zeros(2))
    @test_throws DimensionError ADNLSModel(F, x0, 3, c=c, ncon=1, ucon=zeros(2))
    @test_throws DimensionError ADNLSModel(F, x0, 3, c=c, ncon=1, y0=zeros(2))
    nlp = ADNLSModel(F, x0, 3, -ones(2), ones(2))
    nlp = ADNLSModel(F, x0, 3, c, [-1.0], [1.0])
    nlp = ADNLSModel(F, x0, 3, -ones(2), ones(2), c, [-1.0], [1.0])
  end
end

autodiff_nls_test()
