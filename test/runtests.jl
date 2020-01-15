using MyJuliaGEMM
using LinearAlgebra, Random, Test

Random.seed!(44)

for n in (48, 288, 576)
  A, B, C = rand(n, n), rand(n, n), rand(n, n)
  Cref = C + A * B
  mygemm!(C, A, B)
  @test Cref ≈ C
end
