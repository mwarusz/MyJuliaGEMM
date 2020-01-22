using MyJuliaGEMM

using LinearAlgebra, Printf, Random
BLAS.set_num_threads(Threads.nthreads())

function referencegemm!(C, A, B)
  BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, C)
end

Random.seed!(44)

benchsizes = reverse(48:48:1968)

println(" size   refperf (GFLOP/s)   myperf (GFLOP/s)   errorcheck")
for size in benchsizes
  m, n, k = size, size, size

  gflops = 2e-9 * m * n * k
 
  A    = rand(m, k)
  B    = rand(k, n)
  oldC = rand(m, n)

  refC = Array{Float64}(undef, m, n)
  C = similar(refC)

  referencetime = typemax(Float64)
  for rep = 1:3
    copy!(refC, oldC)
    time = @elapsed referencegemm!(refC, A, B)
    referencetime = min(time, referencetime)
  end
  
  mytime = typemax(Float64)
  for rep = 1:3
    copy!(C, oldC)
    time = @elapsed mygemm!(C, A, B)
    mytime = min(time, mytime)
  end

  referenceperf = gflops / referencetime
  myperf = gflops / mytime
  diff = maximum(abs.(C .- refC))
  
  @printf("%5d %19.2f %18.2f %12.2e\n", n, referenceperf, myperf, diff)
end
