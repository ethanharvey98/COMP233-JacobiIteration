// Ethan Harvey ~ COMP 233 ~ Jacobi Iteration
// Jacobi Iteration for Steady State Heat Distribution in a 2D Plate
// Code by Argonne National Laboratory
// https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAXITERATIONS 500000
#define MAXN 1000

int main() {

	const float north = 100.0;
	const float south = 100.0;
	const float east = 0.0;
	const float west = 0.0;
	const float interiorInit = (north + south + east + west) / 4.0;

	int r, c, i, itcnt;
	float diffnorm;
	float** xold;
	float** xnew;
	float** xtemp;
	// Time variables
	double start, stop, total;

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

	/*
	P3
	1000 1000 # Image width (cols) & height (rows)
	# Joe College ~ COMP322 ~ Laplace Heat Distribution
	# This image took 12387 iterations to converge
	255 # Maximum pixel value
	111 136 176 112 137 177...
	*/

	// Start timer
	start = clock();

	printf("Ethan Harvey ~ COMP 233 ~ Jacobi Iteration\n\n");

	/* Fill the data as specified */
	for (r = 1; r < MAXN; r++) {
		for (c = 0; c < MAXN; c++) {
			xold[r][c] = interiorInit;
		}
		// Set east and west value
		xold[r][MAXN - 1] = east;
		xold[r][0] = west;
		xnew[r][MAXN - 1] = east;
		xnew[r][0] = west;
	}
	for (c = 0; c < MAXN; c++) {
		xold[0][c] = north;
		xold[MAXN - 1][c] = south;
		xnew[0][c] = north;
		xnew[MAXN - 1][c] = south;
	}

	itcnt = 0;
	do {

		/* Compute new values (but not on boundary) */
		itcnt++;
		diffnorm = 0.0;
		for (r = 1; r < MAXN - 1; r++) {
			for (c = 1; c < MAXN - 1; c++) {
				xnew[r][c] = (xold[r][c + 1] + xold[r][c - 1] +
					xold[r + 1][c] + xold[r - 1][c]) / 4.0;
				diffnorm += (xnew[r][c] - xold[r][c]) *
					(xnew[r][c] - xold[r][c]);
			}
		}

		xtemp = xold;
		xold = xnew;
		xnew = xtemp;

		if (itcnt % 1000 == 0) {
			printf("At iteration %d, diff is %e\n", itcnt, sqrt(diffnorm));
		}


	} while (sqrt(diffnorm) > 1.0e-2 && itcnt < MAXITERATIONS);

	// End timer
	stop = clock();
	total = stop - start;

	FILE* fp;
	fp = fopen("jacobi.ppm", "w");
	if (fp != NULL) {
		fprintf(fp, "P3\n%d %d\n", MAXN, MAXN);
		fprintf(fp, "# Ethan Harvey ~ COMP 233 ~ Laplace Heat Distribution\n");
		fprintf(fp, "# This image took %d iterations to converge\n255\n", itcnt);
		for (r = 0; r < MAXN; r++) {
			for (c = 0; c < MAXN; c++) {
				fprintf(fp, "%d 0 %d ", (int)(xnew[r][c] / 100 * 255), (int)(255 - (xnew[r][c] / 100 * 255)));
				if ((c + 1) % 5 == 0) {
					fprintf(fp, "\n");
				}
			}
		}
		fclose(fp);
	}

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