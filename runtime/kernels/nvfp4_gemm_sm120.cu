/*
 * A2 自研 SM120 NVFP4 GEMM kernel — decode-optimized tile configs.
 * Based on vLLM's nvfp4_scaled_mm_sm120_kernels.cu (Apache-2.0).
 *
 * v2: Pre-allocated workspace (CUDA Graph compatible, no per-call cudaMalloc).
 */
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cuda_runtime.h>

using namespace cute;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using ElementAcc = float;
using ArchTag    = cutlass::arch::Sm120;
using OpClass    = cutlass::arch::OpClassBlockScaledTensorOp;

template <typename TileShape, typename Scheduler>
struct BuildGemm {
  using Cluster = Shape<_1,_1,_1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass, TileShape, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, cutlass::layout::RowMajor, 8,
    ElementD, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::layout::RowMajor, 32,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::layout::ColumnMajor, 32,
    ElementAcc, TileShape, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, Main, Epi, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using GemmA = BuildGemm<Shape<_128,_128,_128>, void>::Gemm;
using GemmB = BuildGemm<Shape<_128,_128,_128>, cutlass::gemm::PersistentScheduler>::Gemm;
using GemmC = BuildGemm<Shape<_256,_128,_128>, cutlass::gemm::PersistentScheduler>::Gemm;
using GemmD = BuildGemm<Shape<_128,_256,_128>, cutlass::gemm::PersistentScheduler>::Gemm;

// Pre-allocated workspace (256 KB covers all configs)
static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static void ensure_workspace(size_t needed) {
  if (needed <= g_workspace_size) return;
  if (g_workspace) cudaFree(g_workspace);
  cudaMalloc(&g_workspace, needed);
  g_workspace_size = needed;
}

template <typename Gemm>
static int run_gemm(
    void* D_ptr, const void* A_ptr, const void* B_ptr,
    const void* Asf_ptr, const void* Bsf_ptr, const float* alpha_ptr,
    int M, int N, int K, cudaStream_t stream)
{
  using EA  = typename Gemm::ElementA;
  using EB  = typename Gemm::ElementB;
  using ED  = typename Gemm::ElementD;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BlkCfg  = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ESF = cutlass::float_ue4m3_t;

  auto sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto layout_SFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = BlkCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {reinterpret_cast<EA const*>(A_ptr), sA,
     reinterpret_cast<EB const*>(B_ptr), sB,
     reinterpret_cast<ESF const*>(Asf_ptr), layout_SFA,
     reinterpret_cast<ESF const*>(Bsf_ptr), layout_SFB},
    {{},
     reinterpret_cast<ED const*>(D_ptr), sD,
     reinterpret_cast<ED*>(D_ptr), sD}};
  args.epilogue.thread.alpha_ptr = alpha_ptr;

  Gemm gemm;
  size_t ws_needed = Gemm::get_workspace_size(args);
  ensure_workspace(ws_needed);

  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return (int)st;
  st = gemm.initialize(args, g_workspace, stream);
  if (st != cutlass::Status::kSuccess) return (int)st;
  st = gemm.run(args, g_workspace, stream);
  return (int)st;
}

extern "C" {
int qsr_nvfp4_gemm(
    int config_id,
    void* D, const void* A, const void* B,
    const void* Asf, const void* Bsf, const float* alpha,
    int m, int n, int k, cudaStream_t stream)
{
  switch (config_id) {
    case 0: return run_gemm<GemmA>(D,A,B,Asf,Bsf,alpha,m,n,k,stream);
    case 1: return run_gemm<GemmB>(D,A,B,Asf,Bsf,alpha,m,n,k,stream);
    case 2: return run_gemm<GemmC>(D,A,B,Asf,Bsf,alpha,m,n,k,stream);
    case 3: return run_gemm<GemmD>(D,A,B,Asf,Bsf,alpha,m,n,k,stream);
    default: return -1;
  }
}
}
