using MyJuliaGEMM
using LinearAlgebra, Random, Test

Random.seed!(44)

sizes = (48, 288, 576)
for (k, m, n) in Iterators.product(sizes, sizes, sizes)
  A, B, C = rand(m, k), rand(k, n), rand(m, n)
  Cref = C + A * B
  mygemm!(C, A, B)
  @test Cref ≈ C
end
