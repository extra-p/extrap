#include "Utilities.h"
#include <cmath>
#include <vector>

using namespace std;

void
matrixInverse(double* matrix, int hypothesisTotalTerms)
{
	// convert 1D array to 2D Vector Matrix
	vector<double> line(2*hypothesisTotalTerms,0);
	vector< vector<double> > M(hypothesisTotalTerms,line);
	int index = 0;
	for (int i = 0; i < hypothesisTotalTerms; ++i) {
		for (int j = 0; j < hypothesisTotalTerms; ++j) {
			M[i][j] = matrix[index];
			index++;
		}
	}

	// fill second part of the matrix with 1s
	for (int i=0; i<hypothesisTotalTerms; i++) {
		M[i][hypothesisTotalTerms+i] = 1;
	}

	int n = M.size();

	for (int i=0; i<n; i++) {
		// Search for maximum in this column
		double maxEl = abs(M[i][i]);
		int maxRow = i;
		for (int k=i+1; k<n; k++) {
			if (abs(M[k][i]) > maxEl) {
				maxEl = M[k][i];
				maxRow = k;
			}
		}

		// Swap maximum row with current row (column by column)
		for (int k=i; k<2*n;k++) {
			double tmp = M[maxRow][k];
			M[maxRow][k] = M[i][k];
			M[i][k] = tmp;
		}

		// Make all rows below this one 0 in current column
		for (int k=i+1; k<n; k++) {
			double c = -M[k][i]/M[i][i];
			for (int j=i; j<2*n; j++) {
				if (i==j) {
					M[k][j] = 0;
				} else {
					M[k][j] += c * M[i][j];
				}
			}
		}
	}

	// Solve equation Mx=b for an upper triangular matrix M
	for (int i=n-1; i>=0; i--) {
		for (int k=n; k<2*n;k++) {
			M[i][k] /= M[i][i];
		}
		// this is not necessary, but the output looks nicer:
		M[i][i] = 1;

		for (int rowModify=i-1;rowModify>=0; rowModify--) {
			for (int columModify=n;columModify<2*n;columModify++) {
				M[rowModify][columModify] -= M[i][columModify]
											 * M[rowModify][i];
			}
			// this is not necessary, but the output looks nicer:
			M[rowModify][i] = 0;
		}
	}

	// convert 2D Vector Matrix back to 1D array
	index = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			matrix[index] = M[i][j+n];
			index++;
		}
	}
}