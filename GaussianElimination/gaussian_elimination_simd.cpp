/**
 * Gaussian Elimination with Partial Pivoting — AVX2 SIMD Version
 *
 * Key optimization:
 *   The inner elimination loop  "row[k] -= factor * pivot_row[k]"
 *   is vectorized using AVX2 256-bit registers (__m256d), processing
 *   4 doubles per instruction (FMA: vfnmadd231pd).
 *
 * Storage:
 *   The augmented matrix is stored as a flat, 32-byte-ALIGNED 1-D array
 *   (row-major) so that AVX2 aligned loads (_mm256_load_pd) can be used.
 *   Row stride = ld = next multiple of 4 that is >= (n+1).
 *
 * Build:
 *   g++ -O2 -std=c++17 -mavx2 -mfma -o gaussian_elimination_simd gaussian_elimination_simd.cpp
 *
 * Run:
 *   ./gaussian_elimination_simd
 */

#include <bits/stdc++.h>
#include <immintrin.h>   // AVX2 + FMA intrinsics
using namespace std;

// ─── Constants ──────────────────────────────────────────────────────────────
const double EPS = 1e-9;

// ─── Aligned matrix helpers ──────────────────────────────────────────────────
// We allocate a flat array aligned to 32 bytes.
// ld (leading dimension) is rounded up to the next multiple of 4 so every
// row starts on a 32-byte boundary (4 * sizeof(double) = 32).

struct AlignedMatrix {
    int n;          // system size
    int ld;         // leading dimension (>= n+1, multiple of 4)
    double* data;   // row-major flat array, 32-byte aligned

    AlignedMatrix(int n_)
        : n(n_), ld((n_ + 4) & ~3)   // round n+1 up to multiple of 4
    {
        // posix_memalign on Linux; _aligned_malloc on Windows
#ifdef _WIN32
        data = (double*)_aligned_malloc((size_t)n * ld * sizeof(double), 32);
#else
        if (posix_memalign((void**)&data, 32, (size_t)n * ld * sizeof(double)) != 0)
            data = nullptr;
#endif
        memset(data, 0, (size_t)n * ld * sizeof(double));
    }

    ~AlignedMatrix() {
#ifdef _WIN32
        _aligned_free(data);
#else
        free(data);
#endif
    }

    // Access element [row][col]
    inline double& at(int row, int col) { return data[row * ld + col]; }
    inline double  at(int row, int col) const { return data[row * ld + col]; }

    // Raw pointer to start of a row
    inline double* row_ptr(int row) { return data + row * ld; }
    inline const double* row_ptr(int row) const { return data + row * ld; }
};

// Fill AlignedMatrix from a vector<vector<double>> augmented matrix
AlignedMatrix from_vec(const vector<vector<double>>& A, int n) {
    AlignedMatrix M(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= n; ++j)
            M.at(i, j) = A[i][j];
    return M;
}

// ─── SIMD Core: AVX2 Gaussian Elimination ───────────────────────────────────
/**
 * @brief  Solve Ax = b via partial-pivoting Gaussian elimination.
 *         Inner elimination loop vectorized with AVX2 FMA (4 doubles/cycle).
 * @param  M   AlignedMatrix holding the augmented matrix [A|b]
 * @param  x   Output solution vector (size n)
 * @return true on success; false if singular
 */
bool gaussianElimination_avx2(AlignedMatrix& M, vector<double>& x) {
    const int n  = M.n;
    const int ld = M.ld;

    // ── Forward Elimination ──────────────────────────────────────────────────
    for (int col = 0; col < n; ++col) {

        // 1. Partial pivot
        int    pivotRow = col;
        double maxVal   = fabs(M.at(col, col));
        for (int row = col + 1; row < n; ++row) {
            double v = fabs(M.at(row, col));
            if (v > maxVal) { maxVal = v; pivotRow = row; }
        }
        if (maxVal < EPS) {
            cerr << "[ERROR] Singular matrix detected at column " << col << ".\n";
            return false;
        }

        // 2. Row swap (swap raw pointers with memcpy trick)
        if (pivotRow != col) {
            double* rA = M.row_ptr(col);
            double* rB = M.row_ptr(pivotRow);
            // Swap entire row (including padding) using AVX2 256-bit loads
            int k = 0;
            for (; k + 4 <= ld; k += 4) {
                __m256d a = _mm256_load_pd(rA + k);
                __m256d b = _mm256_load_pd(rB + k);
                _mm256_store_pd(rA + k, b);
                _mm256_store_pd(rB + k, a);
            }
            for (; k < ld; ++k) swap(rA[k], rB[k]);
        }

        // 3. Eliminate rows below pivot
        const double  pivot    = M.at(col, col);
        const double* pivot_rp = M.row_ptr(col);

        // Broadcast pivot_row[col] reciprocal — but we keep per-row factor
        for (int row = col + 1; row < n; ++row) {
            double  factor = M.at(row, col) / pivot;
            double* rr     = M.row_ptr(row);

            // Set eliminated column to exact zero
            rr[col] = 0.0;

            // Vectorized: rr[k] -= factor * pivot_rp[k]  for k = col+1 .. n
            // Use FMA: rr[k] = rr[k] + (-factor) * pivot_rp[k]
            __m256d vfactor = _mm256_set1_pd(-factor);
            int k = col + 1;

            // AVX2 aligned loop (4 doubles at a time)
            for (; k + 4 <= n + 1; k += 4) {
                __m256d vr = _mm256_loadu_pd(rr + k);          // target row (may be unaligned at start)
                __m256d vp = _mm256_loadu_pd(pivot_rp + k);    // pivot row
                vr = _mm256_fmadd_pd(vfactor, vp, vr);         // vr += (-factor)*vp
                _mm256_storeu_pd(rr + k, vr);
            }
            // Scalar tail
            for (; k <= n; ++k)
                rr[k] -= factor * pivot_rp[k];
        }
    }

    // ── Back Substitution ────────────────────────────────────────────────────
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = M.at(i, n);
        for (int j = i + 1; j < n; ++j)
            sum -= M.at(i, j) * x[j];
        x[i] = sum / M.at(i, i);
    }
    return true;
}

// ─── Helper: Print Functions ─────────────────────────────────────────────────
void printSolution(const vector<double>& x, const string& label = "") {
    if (!label.empty()) cout << label << "\n";
    cout << "Solution vector x:\n";
    for (size_t i = 0; i < x.size(); ++i)
        cout << "  x[" << i << "] = " << fixed << setprecision(6) << x[i] << "\n";
    cout << "\n";
}

void verifyResidual(const vector<vector<double>>& A_orig,
                    const vector<double>& x, int n) {
    double maxRes = 0.0;
    for (int i = 0; i < n; ++i) {
        double res = A_orig[i][n];
        for (int j = 0; j < n; ++j) res -= A_orig[i][j] * x[j];
        maxRes = max(maxRes, fabs(res));
    }
    cout << "Residual ||Ax - b||_inf = "
         << scientific << setprecision(3) << maxRes << "\n\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "========== Gaussian Elimination — AVX2 SIMD Version ==========\n\n";

    // ── Example 1: 4x4 system ────────────────────────────────────────────────
    {
        int n = 4;
        vector<vector<double>> Ab = {
            {4, 1, 2, 3, 24},
            {3, 4, 1, 2, 22},
            {2, 3, 4, 1, 24},
            {1, 2, 3, 4, 30}
        };
        // Exact solution: x = [1, 2, 3, 4]

        cout << "--- Example 1: 4x4 System (exact solution: x=[1,2,3,4]) ---\n";
        AlignedMatrix M = from_vec(Ab, n);
        vector<double> x;
        if (gaussianElimination_avx2(M, x)) {
            printSolution(x);
            verifyResidual(Ab, x, n);
        }
    }

    // ── Example 2: 3x3 Hilbert matrix ────────────────────────────────────────
    {
        int n = 3;
        vector<vector<double>> Ab(n, vector<double>(n + 1, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Ab[i][j] = 1.0 / (i + j + 1);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Ab[i][n] += Ab[i][j];   // b = H * [1,1,1]^T

        cout << "--- Example 2: 3x3 Hilbert Matrix (exact solution: x=[1,1,1]) ---\n";
        AlignedMatrix M = from_vec(Ab, n);
        vector<double> x;
        if (gaussianElimination_avx2(M, x)) {
            printSolution(x);
            verifyResidual(Ab, x, n);
        }
    }

    // ── Example 3: Custom input from stdin ───────────────────────────────────
    cout << "--- Example 3: Custom Input ---\n";
    cout << "Enter system size n (0 to skip): ";
    int n; cin >> n;
    if (n > 0) {
        vector<vector<double>> Ab(n, vector<double>(n + 1));
        cout << "Enter augmented matrix [A|b] row by row (" << n + 1 << " values/row):\n";
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= n; ++j)
                cin >> Ab[i][j];
        AlignedMatrix M = from_vec(Ab, n);
        vector<double> x;
        if (gaussianElimination_avx2(M, x)) {
            printSolution(x);
            verifyResidual(Ab, x, n);
        }
    } else {
        cout << "Custom input skipped.\n\n";
    }

    cout << "========== Done ==========\n";
    return 0;
}
