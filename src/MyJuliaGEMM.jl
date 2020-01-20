module MyJuliaGEMM

export mygemm!

using SIMD, StaticArrays
using GPUifyLoops: @unroll

struct BlockSizes{MC, NC, KC, MR, NR} end

function mygemm!(C, A, B; MR = 8, NR = 6, MC = 96, NC = 2016, KC = 256)
  m, n = size(C)

  @assert m % MR == 0
  @assert MC % MR == 0
  @assert n % NR == 0
  @assert NC % NR == 0

  packedA = MVector{MC * KC, Float64}(undef)
  packedB = MVector{KC * NC, Float64}(undef)

  loop_five!(C, A, B, packedA, packedB, BlockSizes{MC, NC, KC, MR, NR}())
end

function loop_five!(C, A, B, packedA, packedB,
                    sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR}
  n = size(C, 2)
  @inbounds for j = 1:NC:n
    jb = min(j + NC - 1, n)
    Ctile = view(C, :, j:jb)
    Btile = view(B, :, j:jb)
    loop_four!(Ctile, A, Btile, packedA, packedB, sizes)
  end
end

@inline function packB!(packedB, B, rangeB,
                        sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  k, n = size(B)
  ix = 1
  @inbounds for jt = 1:NR:n
    @unroll 4 for p in 1:k
      @simd ivdep for j = 0:NR-1
        packedB[ix] = B[p, jt + j]
        ix += 1
      end
    end
  end
end

@inline function loop_four!(C, A, B, packedA, packedB,
                            sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  k = size(A, 2)
  @inbounds for p = 1:KC:k
    pb = min(p + KC - 1, k)
    Atile = view(A, :, p:pb)
    Btile = view(B, p:pb, :)
    packB!(packedB, Btile, p:pb, sizes)
    realKC = (pb - p + 1)
    lineA = realKC * MR
    lineB = realKC * NR
    loop_three!(C, Atile, lineA, packedB, lineB, packedA, sizes)
  end
end

@inline function packA!(packedA, A, rangeA,
                        sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  m, k = size(A)
  ix = 1
  @inbounds for it = 1:MR:m
    @unroll 4 for p = 1:k
      @simd ivdep for i = 0:MR-1
        packedA[ix] = A[it + i, p]
        ix += 1
      end
    end
  end
end

@inline function loop_three!(C, A, lineA, B, lineB, packedA,
                             sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  m = size(C, 1)
  @inbounds for i = 1:MC:m
    ib = min(i + MC - 1, m)
    Ctile = view(C, i:ib, :)
    Atile = view(A, i:ib, :)
    packA!(packedA, Atile, i:ib, sizes)
    loop_two!(Ctile, packedA, lineA, B, lineB, sizes)
  end
end

@inline function loop_two!(C, A, lineA, B, lineB,
                           sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  n = size(C, 2)
  @inbounds for j = 1:NR:n
    Ctile  = view(C, :, j:j+NR-1)
    tix = div(j, NR)
    Btile = view(B, 1+tix*lineB:(tix+1)*lineB)
    loop_one!(Ctile, A, lineA, Btile, sizes)
  end
end

@inline function loop_one!(C, A, lineA, B,
                           sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  m = size(C, 1)
  @inbounds for i = 1:MR:m
    Ctile = view(C, i:i+MR-1, :)
    tix = div(i, MR)
    Atile = view(A, 1+tix*lineA:(tix+1)*lineA)
    micro_kernel!(Ctile, Atile, B, sizes)
  end
end

@inline function micro_kernel!(C, A, B,
                               sizes::BlockSizes{MC, NC, KC, MR, NR}) where {MC, NC, KC, MR, NR} 
  vecT = Vec{4, Float64}
  MRdiv4 = div(MR, 4)
  # full C size
  m = size(parent(C), 1)

  cT = MArray{Tuple{MRdiv4, NR}, vecT}(undef)

  @unroll for q = 1:NR
    @unroll for r = 1:MRdiv4
      offset = 8 * (4(r - 1) + (q - 1) * m)
      @inbounds cT[r, q] = vload(vecT, pointer(C) + offset, Val(true))
    end
  end

  a = MVector{MRdiv4, vecT}(undef)

  lBdiv4 = div(length(B), NR) 

  @inbounds @unroll 4 for p = 1:lBdiv4
    @unroll for r = 1:MRdiv4
      offset = 8 * (4(r - 1) + MR * (p - 1))
      a[r] = vload(vecT, pointer(A) + offset, Val(true))
    end

    @unroll for q = 1:NR
      b = vecT(B[q + NR * (p - 1)])
      @unroll for r = 1:MRdiv4
        cT[r, q] = muladd(a[r], b, cT[r, q])
      end
    end
  end

  @unroll for q = 1:NR
    @unroll for r = 1:MRdiv4
      offset = 8 * (4(r - 1) + (q - 1) * m)
      @inbounds vstore(cT[r, q], pointer(C) + offset, Val(true), Val(true))
    end
  end
end

end # module
