/**
 * Gaussian Elimination with Partial Pivoting
 *
 * Features:
 *   - Solves n x n augmented matrix systems Ax = b
 *   - Uses partial pivoting for improved numerical stability
 *
 * Build:
 *   g++ -O2 -std=c++17 -o gaussian_elimination gaussian_elimination.cpp
 *
 * Run:
 *   ./gaussian_elimination
 */

#include <bits/stdc++.h>
using namespace std;

// ─── Constants ──────────────────────────────────────────────────────────────
const double EPS = 1e-9;  // threshold for near-zero pivot detection

// ─── Core: Gaussian Elimination ─────────────────────────────────────────────
/**
 * @brief  Solve Ax = b via partial-pivoting Gaussian elimination + back-sub.
 * @param  A  Augmented matrix [A|b] of size n x (n+1), 0-indexed
 * @param  n  System size
 * @param  x  Output solution vector
 * @return true if a unique solution exists; false if the matrix is singular
 */
bool gaussianElimination(vector<vector<double>>& A, int n, vector<double>& x) {
    // ── Forward Elimination ───────────────────────────────────────────────────
    for (int col = 0; col < n; ++col) {
        // 1. Partial pivot: find the row with the largest absolute value in this column
        int pivotRow = col;
        double maxVal = fabs(A[col][col]);
        for (int row = col + 1; row < n; ++row) {
            if (fabs(A[row][col]) > maxVal) {
                maxVal = fabs(A[row][col]);
                pivotRow = row;
            }
        }

        // 2. Singular check
        if (maxVal < EPS) {
            cerr << "[ERROR] Singular matrix: the system may have no solution or infinitely many solutions.\n";
            return false;
        }

        // 3. Swap current row with pivot row
        if (pivotRow != col) {
            swap(A[col], A[pivotRow]);
        }

        // 4. Eliminate entries below the pivot
        double pivot = A[col][col];
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row][col] / pivot;
            // Update from col onward (SIMD-friendly layout)
            for (int k = col; k <= n; ++k) {
                A[row][k] -= factor * A[col][k];
            }
            A[row][col] = 0.0;  // eliminate floating-point accumulation
        }
    }

    // ── Back Substitution ────────────────────────────────────────────────────
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = A[i][n];  // right-hand side b_i
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return true;
}

// ─── Helper: Print Functions ────────────────────────────────────────────────
void printMatrix(const vector<vector<double>>& A, const string& title) {
    cout << title << "\n";
    for (const auto& row : A) {
        cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            cout << setw(10) << fixed << setprecision(4) << row[j];
            if (j + 1 < row.size()) cout << "  ";
        }
        cout << " ]\n";
    }
    cout << "\n";
}

void printSolution(const vector<double>& x) {
    cout << "Solution vector x:\n";
    for (size_t i = 0; i < x.size(); ++i) {
        cout << "  x[" << i << "] = " << fixed << setprecision(6) << x[i] << "\n";
    }
    cout << "\n";
}

// ─── Helper: Residual Verification ─────────────────────────────────────────
/**
 * @brief Compute and print the residual ||Ax - b||_inf to verify accuracy.
 */
void verifyResidual(const vector<vector<double>>& A_orig,
                    const vector<double>& x, int n) {
    double maxRes = 0.0;
    for (int i = 0; i < n; ++i) {
        double res = A_orig[i][n];  // b_i
        for (int j = 0; j < n; ++j) {
            res -= A_orig[i][j] * x[j];
        }
        maxRes = max(maxRes, fabs(res));
    }
    cout << "Residual ||Ax - b||_inf = " << scientific << setprecision(3) << maxRes << "\n\n";
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "========== Gaussian Elimination with Partial Pivoting ==========\n\n";

    // ── Example 1: 4x4 system ────────────────────────────────────────────────
    {
        int n = 4;
        // Augmented matrix [A|b]; last column is the right-hand side b.
        // Exact solution: x0=1, x1=2, x2=3, x3=4
        //   row0:  4x0 + 1x1 + 2x2 + 3x3 = 4+2+6+12 = 24
        //   row1:  3x0 + 4x1 + 1x2 + 2x3 = 3+8+3+8  = 22
        //   row2:  2x0 + 3x1 + 4x2 + 1x3 = 2+6+12+4 = 24
        //   row3:  1x0 + 2x1 + 3x2 + 4x3 = 1+4+9+16 = 30
        vector<vector<double>> A = {
            {4, 1, 2, 3, 24},
            {3, 4, 1, 2, 22},
            {2, 3, 4, 1, 24},
            {1, 2, 3, 4, 30}
        };
        vector<vector<double>> A_orig = A;  // keep original for residual check

        cout << "--- Example 1: 4x4 System ---\n";
        printMatrix(A_orig, "Augmented matrix [A|b] (original):");

        vector<double> x;
        if (gaussianElimination(A, n, x)) {
            printSolution(x);
            verifyResidual(A_orig, x, n);
        }
    }

    // ── Example 2: 3x3 Hilbert (ill-conditioned) matrix ────────────────────
    {
        int n = 3;
        // Hilbert matrix H_ij = 1/(i+j+1); b = H * [1,1,...,1]^T
        vector<vector<double>> A(n, vector<double>(n + 1, 0.0));
        vector<double> exact(n, 1.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = 1.0 / (i + j + 1);
            }
        }
        // Construct b so that the exact solution is all ones
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][n] += A[i][j] * exact[j];
            }
        }
        vector<vector<double>> A_orig = A;

        cout << "--- Example 2: 3x3 Hilbert Matrix (ill-conditioned) ---\n";
        printMatrix(A_orig, "Augmented matrix [A|b] (original):");

        vector<double> x;
        if (gaussianElimination(A, n, x)) {
            printSolution(x);
            verifyResidual(A_orig, x, n);
        }
    }

    // ── Example 3: Custom input from stdin ──────────────────────────────────
    cout << "--- Example 3: Custom Input ---\n";
    cout << "Enter system size n (enter 0 to skip): ";
    int n;
    cin >> n;
    if (n > 0) {
        vector<vector<double>> A(n, vector<double>(n + 1));
        cout << "Enter the augmented matrix [A|b] row by row (" << n + 1 << " values per row):\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= n; ++j) {
                cin >> A[i][j];
            }
        }
        vector<vector<double>> A_orig = A;
        printMatrix(A_orig, "Augmented matrix [A|b] (original):");

        vector<double> x;
        if (gaussianElimination(A, n, x)) {
            printSolution(x);
            verifyResidual(A_orig, x, n);
        }
    } else {
        cout << "Custom input skipped.\n\n";
    }

    cout << "========== Done ==========\n";
    return 0;
}
