// Ethan Harvey ~ COMP 233 ~ Jacobi Iteration
// Jacobi Iteration for Steady State Heat Distribution in a 2D Plate
// Code by Argonne National Laboratory
// https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {

	// Max number of iterations
	const int MAXITERATIONS = 500000;
	// Dimensions of the 2D array
	const int MAXN = 1000;
	// Starting north value
	const float NORTH = 100.0;
	// Starting south value
	const float SOUTH = 100.0;
	// Starting east value
	const float EAST = 0.0;
	// Starting west value
	const float WEST = 0.0;
	// Starting interior value
	const float INTERIOR = (NORTH + SOUTH + EAST + WEST) / 4.0;

	// Loop variables
	int r, c, i, itcnt;
	// Differential
	float diffnorm;
	// Old 2D array
	float** xold;
	// New 2D array
	float** xnew;
	// Temp pointer
	float** xtemp;
	// Time variables
	clock_t start, stop;
	double total;

	// Start timer
	start = clock();

	// xold
	xold = (float**)malloc((MAXN) * sizeof(float*));
	for (i = 0; i < MAXN; i++) {
		xold[i] = (float*)malloc((MAXN) * sizeof(float));
	}

	// xnew
	xnew = (float**)malloc((MAXN) * sizeof(float*));
	for (i = 0; i < MAXN; i++) {
		xnew[i] = (float*)malloc((MAXN) * sizeof(float));
	}

	// Start header
	printf("Ethan Harvey ~ COMP 233 ~ Jacobi Iteration\n\n");

	// Fill the 2D array
	for (r = 1; r < MAXN; r++) {
		for (c = 0; c < MAXN; c++) {
			xold[r][c] = INTERIOR;
		}
		// Set east and west values
		xold[r][MAXN - 1] = EAST;
		xold[r][0] = WEST;
		xnew[r][MAXN - 1] = EAST;
		xnew[r][0] = WEST;
	}
	for (c = 0; c < MAXN; c++) {
		// Set north and south values
		xold[0][c] = NORTH;
		xold[MAXN - 1][c] = NORTH;
		xnew[0][c] = SOUTH;
		xnew[MAXN - 1][c] = SOUTH;
	}

	itcnt = 0;
	do {

		// Compute new values but not on boundary
		itcnt++;
		diffnorm = 0.0;
		for (r = 1; r < MAXN - 1; r++) {
			for (c = 1; c < MAXN - 1; c++) {
				xnew[r][c] = (xold[r][c + 1] + xold[r][c - 1] +
					xold[r + 1][c] + xold[r - 1][c]) / 4.0;
				diffnorm += (xnew[r][c] - xold[r][c]) *
					(xnew[r][c] - xold[r][c]);
			}
		} // End loop

		// Swap pointers
		xtemp = xold;
		xold = xnew;
		xnew = xtemp;

		if (itcnt % 1000 == 0) {
			printf("At iteration %d, diff is %e\n", itcnt, sqrt(diffnorm));
		}


	} while (sqrt(diffnorm) > 1.0e-2 && itcnt < MAXITERATIONS);

	// Print to file
	FILE* fp;
	fp = fopen("jacobi.ppm", "w");
	if (fp != NULL) {
		// Print header to file
		fprintf(fp, "P3\n%d %d\n", MAXN, MAXN);
		fprintf(fp, "# Ethan Harvey ~ COMP 233 ~ Laplace Heat Distribution\n");
		fprintf(fp, "# This image took %d iterations to converge\n255\n", itcnt);
		for (r = 0; r < MAXN; r++) {
			for (c = 0; c < MAXN; c++) {
				// Calculate each pixel value
				fprintf(fp, "%d 0 %d ", (int)(xnew[r][c] / 100 * 255), (int)(255 - (xnew[r][c] / 100 * 255)));
				if ((c + 1) % 5 == 0) {
					fprintf(fp, "\n");
				}
			}
		}
		fclose(fp);
	}

	// End timer
	stop = clock();
	total = (double) (stop - start) / CLOCKS_PER_SEC;
	
	// Cleanup
	for (i = 0; i < MAXN; i++) {
		free(xold[i]);
	}
	free(xold);
	for (i = 0; i < MAXN; i++) {
		free(xnew[i]);
	}
	free(xnew);

	// Normal Termination
	printf("Total time: %f\n", total);
	printf("\n\n<<< Normal Termination >>>\n\n");

	return 0;
}