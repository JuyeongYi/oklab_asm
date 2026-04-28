#!/usr/bin/env bash
# build.sh — 모든 구현체와 테스트/벤치를 한 번에 빌드.
#
# 사용:
#   ./build.sh              # CPU 전체 빌드 + 테스트
#   ./build.sh master       # CPU master_bench 빌드 후 실행
#   ./build.sh cuda         # CUDA 포함 master_all + cuda_bench 빌드 후 실행
#   ./build.sh all          # CPU + CUDA 둘 다
#   ./build.sh clean        # 빌드 산출물 삭제

set -e

CXX=${CXX:-g++}
NVCC=${NVCC:-nvcc}
NVCC_ARCH=${NVCC_ARCH:-sm_89}     # Ada Lovelace 기본.  Ampere = sm_80, Hopper = sm_90 등.

STD="-std=c++17 -O3 -DNDEBUG -flto"
SSE_FLAGS="-msse4.1 -mfma"
AVX512_FLAGS="-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1"
COMMON_FLAGS="-Wall -Wextra"

INCLUDE_FLAGS="-Ioriginal -Iscalar_vec3 -Iscalar_vec4 -Isimd_sse \
               -Isimd_avx512_vl -Isimd_avx512 -Ihybrid -Icuda"

ALL_IMPLS_CPP="scalar_vec3/ok_color.cpp \
               scalar_vec4/ok_color_v4.cpp \
               simd_sse/ok_color_simd.cpp \
               simd_avx512_vl/ok_color_avx512_4.cpp \
               simd_avx512/ok_color_avx512.cpp \
               hybrid/ok_color_best_asm.cpp"

mkdir -p build

build_cpu() {
	local target=$1
	local src=$2
	echo "[g++] $target"
	$CXX $STD $AVX512_FLAGS $COMMON_FLAGS $INCLUDE_FLAGS \
		$ALL_IMPLS_CPP $src -o build/$target
}

# nvcc 로 CUDA 빌드.  CPU .cpp 들은 -Xcompiler 로 g++ 에 AVX512 플래그 전달.
build_cuda() {
	local target=$1
	local src=$2
	echo "[nvcc] $target"
	$NVCC -O3 -arch=$NVCC_ARCH -std=c++17 \
		-Xcompiler "$AVX512_FLAGS -O3 -DNDEBUG" \
		$INCLUDE_FLAGS \
		$ALL_IMPLS_CPP cuda/ok_color_cuda.cu \
		$src -o build/$target
}

run_target() {
	local target=$1
	echo
	echo "============================================"
	echo "  $target"
	echo "============================================"
	./build/$target
}

case "${1:-cpu}" in
	clean)
		rm -rf build
		echo "[clean] build/ removed"
		;;
	master)
		build_cpu master_bench benches/master_bench.cpp
		run_target master_bench
		;;
	cuda)
		build_cuda master_all benches/master_all.cu
		build_cuda cuda_bench cuda/cuda_bench.cu
		run_target cuda_bench
		run_target master_all
		;;
	ptx)
		echo "[nvcc] ptx_full"
		$NVCC -O3 -arch=$NVCC_ARCH -std=c++17 -Icuda \
			cuda/ok_color_cuda.cu ptx/ok_color_ptx_full.cu -o build/ptx_full
		run_target ptx_full
		;;
	tensor)
		echo "[nvcc] tensor_bench"
		$NVCC -O3 -arch=$NVCC_ARCH -std=c++17 -Icuda \
			cuda/ok_color_cuda.cu tensor/ok_color_tensor.cu -o build/tensor_bench
		run_target tensor_bench
		;;
	tests)
		build_cpu compare_test    tests/compare_test.cpp
		build_cpu simd_verify     tests/simd_verify.cpp
		build_cpu best_asm_verify tests/best_asm_verify.cpp
		build_cpu smoke_test      tests/smoke_test.cpp
		run_target compare_test
		run_target simd_verify
		run_target best_asm_verify
		run_target smoke_test
		;;
	benches)
		build_cpu master_bench benches/master_bench.cpp
		run_target master_bench
		;;
	cpu|all-cpu)
		build_cpu compare_test    tests/compare_test.cpp
		build_cpu simd_verify     tests/simd_verify.cpp
		build_cpu best_asm_verify tests/best_asm_verify.cpp
		build_cpu master_bench    benches/master_bench.cpp
		run_target compare_test
		run_target simd_verify
		run_target best_asm_verify
		run_target master_bench
		;;
	all)
		build_cpu compare_test    tests/compare_test.cpp
		build_cpu simd_verify     tests/simd_verify.cpp
		build_cpu best_asm_verify tests/best_asm_verify.cpp
		build_cpu master_bench    benches/master_bench.cpp
		build_cuda cuda_bench     cuda/cuda_bench.cu
		build_cuda master_all     benches/master_all.cu
		run_target compare_test
		run_target simd_verify
		run_target best_asm_verify
		run_target master_bench
		run_target cuda_bench
		run_target master_all
		;;
	*)
		echo "Usage: $0 [cpu|master|cuda|tests|benches|all|clean]"
		exit 1
		;;
esac
