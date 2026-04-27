#!/usr/bin/env bash
# build.sh — 모든 구현체와 테스트/벤치를 한 번에 빌드.
#
# 사용:
#   ./build.sh              # 전체 빌드 + 테스트
#   ./build.sh master       # master_bench 만 빌드 후 실행
#   ./build.sh clean        # 빌드 산출물 삭제

set -e

CXX=${CXX:-g++}
STD="-std=c++17 -O3 -DNDEBUG -flto"
SSE_FLAGS="-msse4.1 -mfma"
AVX512_FLAGS="-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1"
COMMON_FLAGS="-Wall -Wextra"

INCLUDE_FLAGS="-Ioriginal -Iscalar_vec3 -Iscalar_vec4 -Isimd_sse \
               -Isimd_avx512_vl -Isimd_avx512 -Ihybrid"

# 모든 구현체 .cpp 목록 (link 시 함께 사용).
ALL_IMPLS="scalar_vec3/ok_color.cpp \
           scalar_vec4/ok_color_v4.cpp \
           simd_sse/ok_color_simd.cpp \
           simd_avx512_vl/ok_color_avx512_4.cpp \
           simd_avx512/ok_color_avx512.cpp \
           hybrid/ok_color_best_asm.cpp"

mkdir -p build

build_target() {
	local target=$1
	local src=$2
	echo "[build] $target"
	$CXX $STD $AVX512_FLAGS $COMMON_FLAGS $INCLUDE_FLAGS \
		$ALL_IMPLS $src -o build/$target
}

run_target() {
	local target=$1
	echo
	echo "============================================"
	echo "  $target"
	echo "============================================"
	./build/$target
}

case "${1:-all}" in
	clean)
		rm -rf build
		echo "[clean] build/ removed"
		;;
	master)
		build_target master_bench benches/master_bench.cpp
		run_target master_bench
		;;
	tests)
		build_target compare_test    tests/compare_test.cpp
		build_target simd_verify     tests/simd_verify.cpp
		build_target best_asm_verify tests/best_asm_verify.cpp
		build_target smoke_test      tests/smoke_test.cpp
		run_target compare_test
		run_target simd_verify
		run_target best_asm_verify
		run_target smoke_test
		;;
	benches)
		build_target bench               benches/bench.cpp
		build_target bench_three         benches/bench_three.cpp
		build_target bench_cpp_sse_avx512 benches/bench_cpp_sse_avx512.cpp
		build_target bench_hybrid        benches/bench_hybrid.cpp
		build_target bench_four          benches/bench_four.cpp
		build_target master_bench        benches/master_bench.cpp
		run_target master_bench
		;;
	all)
		build_target compare_test    tests/compare_test.cpp
		build_target simd_verify     tests/simd_verify.cpp
		build_target best_asm_verify tests/best_asm_verify.cpp
		build_target master_bench    benches/master_bench.cpp
		echo
		echo "[run] correctness tests"
		run_target compare_test
		run_target simd_verify
		run_target best_asm_verify
		run_target master_bench
		;;
	*)
		echo "Usage: $0 [all|master|tests|benches|clean]"
		exit 1
		;;
esac
